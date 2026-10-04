"""Evaluator-only held-out L1 scoring, called after the external study freeze.

No training imports, checkpoint selection, or test-driven model changes occur here.
The supplied three branches must be greedy outputs from the same locked seed.
"""
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from statistics import mean
import json
import math

from scipy.stats import beta
from bcs.calibration import BooleanCore
from bcs.contracts import Identification, Computation, Grounding
from bcs.core_evaluation import compare_kernels
from bcs.dataset import audit_dataset
from bcs.dataset_codec import event_from_wire
from bcs.generator import Executed, compile_history, event_wire
from bcs.inference import (evaluate, ExactDistribution, Predictive, Interventional,
                           Counterfactual, InferenceUndefined, InferenceIncomplete)
from bcs.language_evaluation import structural_report
from bcs.language_tokens import encode_events, decode_events, InvalidDecoding
from bcs.metrics import empirical_bernstein_upper
from bcs.pairs import PairKind, HoldoutStratum
from bcs.simulator import WorldSpec, Regime, Step, Clamp, Action, Variable as V

BRANCHES = ('concat', 'attention', 'gru')
CATEGORIES = ('predict_hold', 'do_c_zero_hold', 'do_c_one_hold', 'replace_past_command')
STRATA = tuple(f'{kind.value}/{stratum.value}' for kind in PairKind for stratum in HoldoutStratum)
SIMULATOR = WorldSpec.for_regime(Regime.DETERMINISTIC)


def clopper_pearson_upper(errors, groups, alpha=.05 / 14):
    """One-sided exact binomial upper bound (not the two-sided alpha/2 bound)."""
    if type(errors) is not int or type(groups) is not int or not 0 <= errors <= groups or groups < 1 or not 0 < alpha < 1:
        raise ValueError('invalid binomial sample')
    if errors == groups:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / groups)
    return float(beta.ppf(1 - alpha, errors + 1, groups - errors))


def _upper(values, lower=0, upper=1):
    return empirical_bernstein_upper(values, lower, upper) if len(values) >= 2 else None


def paired_excess(branch, concat):
    """Pair by exact independent group IDs; input order has no statistical meaning."""
    if not branch or set(branch) != set(concat):
        raise ValueError('paired group ID sets must match exactly')
    differences = [branch[group] - concat[group] for group in sorted(branch)]
    return {'groups': len(differences), 'mean': mean(differences),
            'upper': _upper(differences, -1, 1), 'range': [-1, 1], 'alpha': .05,
            'group_differences': {group: branch[group] - concat[group] for group in sorted(branch)}}


def _query_panels(events):
    categories = [tuple(Predictive(obj, (Step.hold(),), (V.M, V.Y)) for obj in (0, 1))]
    categories += [tuple(Interventional(obj, (Clamp(obj, V.C, c),), (V.M, V.Y)) for obj in (0, 1)) for c in (0, 1)]
    categories.append(tuple(Counterfactual(e.object_id, e.time,
        Action.STOP if e.action == Action.START else Action.START, (V.M, V.Y))
        for e in events if isinstance(e, Executed)))
    if not categories[-1]:
        raise ValueError('gold history lacks an eligible executed CF reference')
    joint = tuple(Interventional(obj, tuple(Clamp(obj, variable, value)
        for variable, value in zip(targets, assignment)), (V.M, V.Y))
        for obj in (0, 1) for targets in combinations((V.R, V.M, V.C), 2)
        for assignment in product((0, 1), repeat=2))
    return tuple(categories), joint


def _answer(core, history, query, cache):
    if history is None:
        return InferenceUndefined('invalid grounding')
    key = (getattr(core, 'artifact_id', id(core)), history, query)
    if key not in cache:
        try:
            cache[key] = evaluate(core, history, query)
        except ValueError as error:
            cache[key] = InferenceUndefined(str(error))
    return cache[key]


def _distance(a, b):
    if not isinstance(a, ExactDistribution) or not isinstance(b, ExactDistribution):
        return 1.0
    p, q = dict(a.probabilities), dict(b.probabilities)
    return float(sum(abs(p.get(outcome, 0) - q.get(outcome, 0))
                     for outcome in p.keys() | q.keys()) / 2)


def _status(answer, invalid=False):
    # A conditional law under a fixed SCM is not an identification certificate.
    # Result-kind is the exact joint-law variant, not contracts.Point's scalar.
    return (Identification.NOT_ESTABLISHED.value,
            Computation.NOT_STARTED.value if invalid else Computation.BUDGET.value if isinstance(answer, InferenceIncomplete) else Computation.COMPLETE.value,
            Grounding.INVALID.value if invalid else Grounding.RESOLVED.value,
            type(answer).__name__)


def score_history(tokens, events, core, cache=None):
    """All μ_lang/μ_joint comparisons for a single visible gold history."""
    cache = {} if cache is None else cache
    decoded = decode_events(tokens)
    invalid = isinstance(decoded, InvalidDecoding)
    predicted = None if invalid else compile_history(decoded.events)
    truth = compile_history(events)
    categories, joint = _query_panels(events)
    panel = tuple(query for category in categories for query in category) + joint
    answers = {source: [_answer(model, history, query, cache) for query in panel]
               for source, model, history in (('branch', core, predicted), ('oracle', core, truth),
                                              ('simulator', SIMULATOR, truth))}
    if any(not isinstance(answer, ExactDistribution) for answer in answers['simulator']):
        raise ValueError('gold simulator point query is undefined')
    result = {}
    for left, right in (('branch', 'oracle'), ('branch', 'simulator'), ('oracle', 'simulator')):
        distances = [_distance(a, b) for a, b in zip(answers[left], answers[right])]
        offset = 0
        values = []
        for category in categories:
            values.append(mean(distances[offset:offset + len(category)]))
            offset += len(category)
        result[f'{left}_to_{right}'] = {'categories': values, 'language': mean(values),
                                      'joint': mean(distances[offset:])}
    result['coverage'] = {'invalid_grounding': invalid, 'queries': len(panel),
        'point_queries': sum(isinstance(answer, ExactDistribution) for answer in answers['branch']),
        'point_fraction': mean(isinstance(answer, ExactDistribution) for answer in answers['branch']),
        'statuses': dict(Counter('/'.join(_status(answer, invalid)) for answer in answers['branch']))}
    # Not persisted: these laws permit exact cross-paraphrase comparisons on the
    # same query panel without running inference again.
    result['_language_answers'] = [answers['branch'][sum(map(len, categories[:i])):
        sum(map(len, categories[:i + 1]))] for i in range(4)]
    result['_invalid'] = invalid
    return result


def _canonical(events):
    return tuple(sorted(events, key=lambda event: (event.time,
        json.dumps(event_wire(event), sort_keys=True, separators=(',', ':')))))


def structural_scores(predictions, targets, normalize):
    """Field confusions plus complete interpretation checks; preserves multiplicity."""
    if not (len(predictions) == len(targets) == len(normalize)) or not targets:
        raise ValueError('nonempty aligned structural samples required')
    normalized_predictions, normalized_targets, critical = [], [], []
    for tokens, events, invariant in zip(predictions, targets, normalize):
        decoded = decode_events(tokens)
        gold = _canonical(events) if invariant else tuple(events)
        normalized_targets.append(encode_events(gold))
        if isinstance(decoded, InvalidDecoding):
            normalized_predictions.append(tuple(tokens))
            critical.append(1)
        else:
            predicted = _canonical(decoded.events) if invariant else decoded.events
            normalized_predictions.append(encode_events(predicted))
            critical.append(int(predicted != gold))
    report = structural_report(normalized_predictions, normalized_targets)
    for field in report['fields'].values():
        field['exact_match'] = 1 - field['error_rate']
    report.update(full_event_exact_match=1 - report['full_event_error'],
                  full_history_exact_match=1 - report['full_history_error'], critical_errors=critical,
                  alignment='canonical timestamp/event rank for invariant pairs; narrative positions otherwise',
                  critical_definition='any error in complete checked interpretation on either side; all six fields and context retained',
                  intervals='per-field descriptive only; no message-level independence assumption')
    return report


def _summary(values):
    if not values:
        return {'groups': 0, 'language_mean_tv': None, 'language_tv_upper': None,
                'joint_mean_tv': None, 'joint_tv_upper': None, 'category_means': {key: None for key in CATEGORIES}}
    language, joint = [v['language'] for v in values], [v['joint'] for v in values]
    cats = [mean(v['categories'][i] for v in values) for i in range(4)]
    return {'groups': len(values), 'category_means': dict(zip(CATEGORIES, cats)),
            'language_mean_tv': mean(language), 'language_tv_upper': _upper(language),
            'joint_mean_tv': mean(joint), 'joint_tv_upper': _upper(joint), 'alpha': .05,
            'sensitivity': {name: sum(x * w for x, w in zip(cats, weights)) for name, weights in
                (('intervention_priority', (.1, .35, .35, .2)), ('counterfactual_priority', (.1, .2, .2, .5)))}}


def _group_comparison(scores, comparison):
    return {'categories': [mean(score[comparison]['categories'][i] for score in scores) for i in range(4)],
            'language': mean(score[comparison]['language'] for score in scores),
            'joint': mean(score[comparison]['joint'] for score in scores)}


def _gate(registered, bounds, threshold):
    if not registered:
        return 'not_run_registered_dataset'
    if any(bound is None for bound in bounds):
        return 'invalid_insufficient_groups'
    return 'passed' if max(bounds) <= threshold else 'failed'


def evaluate_seed(dataset: Path, predictions: dict[str, dict[str, tuple[int, ...]]], output: Path) -> dict:
    """Score held-out labels only after caller has frozen the whole final study.

    This function cannot establish the external freeze, test-loader boundary,
    analytic/status fixtures, or deterministic replay; those remain separate
    evidence gates in the orchestrator. It never accepts supplied oracle outputs.
    """
    dataset, output = Path(dataset), Path(output)
    if set(predictions) != set(BRANCHES):
        raise ValueError('exactly concat, attention, gru branches required')
    audit = audit_dataset(dataset)
    if audit['integrity_status'] != 'passed':
        raise ValueError(f'invalid dataset: {audit["errors"]}')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    core = BooleanCore.from_wire(json.loads((dataset / 'core.json').read_text()))
    rows = {split: [json.loads(line) for line in (dataset / f'evaluator/{split}.jsonl').read_text().splitlines()]
            for split in ('iid_test', 'challenge')}
    record_ids, group_ids = set(), set()
    for split, split_rows in rows.items():
        for row in split_rows:
            group = row['group_id']
            if group in group_ids or len(row['sides']) != (1 if split == 'iid_test' else 2):
                raise ValueError('duplicate group or incorrect side count')
            group_ids.add(group)
            for index, side in enumerate(row['sides']):
                rid = side['record_id']
                if rid in record_ids or rid != f'{group}:{index}':
                    raise ValueError('duplicate or misaligned record ID')
                record_ids.add(rid)
    for branch in BRANCHES:
        if set(predictions[branch]) != record_ids:
            raise ValueError(f'{branch}: prediction record ID set must match IID and challenge exactly')
        if any(not isinstance(tokens, (tuple, list)) for tokens in predictions[branch].values()):
            raise ValueError('predictions must be structured token sequences')
    counts = Counter(f'{row["kind"]}/{row["stratum"]}' for row in rows['challenge'])
    registered = (manifest['mode'] == 'registered_size' and
                  manifest['counts'] == {'train': 20000, 'validation': 4000, 'iid_test': 4000, 'pairs_per_stratum': 1000}
                  and len(rows['iid_test']) == 4000 and set(counts) == set(STRATA)
                  and all(counts[stratum] == 1000 for stratum in STRATA))
    panels = {'iid': [], 'challenge': []}
    structure = {branch: {'predictions': [], 'targets': [], 'normalize': [], 'groups': []} for branch in BRANCHES}
    paraphrase = {branch: defaultdict(list) for branch in BRANCHES}
    coverage = {branch: Counter() for branch in BRANCHES}
    status_counts = {branch: Counter() for branch in BRANCHES}
    for split, split_rows in rows.items():
        for row in split_rows:
            group = {'group_id': row['group_id'], 'kind': row['kind'], 'stratum': row['stratum'], 'comparisons': {}}
            cache = {}
            for branch in BRANCHES:
                scores = []
                errors = []
                for side in row['sides']:
                    events = tuple(event_from_wire(event) for event in side['events'])
                    tokens = predictions[branch][side['record_id']]
                    score = score_history(tokens, events, core, cache)
                    scores.append(score)
                    normalized = row['kind'] in ('paraphrase', 'time')
                    errors += structural_scores([tokens], [events], [normalized])['critical_errors']
                    st = structure[branch]
                    st['predictions'].append(tokens); st['targets'].append(events); st['normalize'].append(normalized); st['groups'].append((split, row['kind'], row['stratum']))
                    coverage[branch].update({key: score['coverage'][key] for key in ('queries', 'point_queries', 'invalid_grounding')})
                    status_counts[branch].update(score['coverage']['statuses'])
                group.setdefault('critical_errors', {})[branch] = int(any(errors))
                for comparison in ('branch_to_oracle', 'branch_to_simulator', 'oracle_to_simulator'):
                    name = comparison.replace('branch', branch)
                    group['comparisons'][name] = _group_comparison(scores, comparison)
                if row['kind'] == 'paraphrase':
                    a, b = scores
                    law_a, law_b = a['_language_answers'], b['_language_answers']
                    tv = mean(mean(_distance(x, y) for x, y in zip(xs, ys)) for xs, ys in zip(law_a, law_b))
                    disagreement = int(any(_status(x, a['_invalid']) != _status(y, b['_invalid'])
                        for xs, ys in zip(law_a, law_b) for x, y in zip(xs, ys)))
                    paraphrase[branch][row['stratum']].append({'group_id': row['group_id'], 'tv': tv, 'status_disagreement': disagreement})
            panels['iid' if split == 'iid_test' else 'challenge'].append(group)
    comparison_names = ('oracle_to_simulator',) + tuple(f'{branch}_to_{target}' for branch in BRANCHES for target in ('oracle', 'simulator'))
    iid = {'groups': panels['iid'], 'comparisons': {name: _summary([group['comparisons'][name] for group in panels['iid']]) for name in comparison_names}}
    challenge = {'groups': panels['challenge'], 'strata': {stratum: {'groups': counts[stratum],
        'comparisons': {name: _summary([group['comparisons'][name] for group in panels['challenge']
            if f'{group["kind"]}/{group["stratum"]}' == stratum]) for name in comparison_names}} for stratum in STRATA},
        'pooled_mean_interval': 'not_computed_fixed_strata'}
    branches = {}
    for branch in BRANCHES:
        critical = {}
        for stratum in STRATA:
            values = [group['critical_errors'][branch] for group in panels['challenge'] if f'{group["kind"]}/{group["stratum"]}' == stratum]
            n, errors = len(values), sum(values)
            upper = clopper_pearson_upper(errors, n) if n else None
            critical[stratum] = {'groups': n, 'errors': errors, 'rate': errors / n if n else None,
                'upper': upper, 'alpha': .05 / 14, 'gate': _gate(registered, [upper], .01)}
        para = {}
        for stratum in HoldoutStratum:
            values = paraphrase[branch][stratum.value]
            n, errors = len(values), sum(value['status_disagreement'] for value in values)
            tv = [value['tv'] for value in values]
            disagreement_upper = clopper_pearson_upper(errors, n) if n else None
            para[stratum.value] = {'groups': n, 'status_disagreements': errors,
                'status_disagreement_rate': errors / n if n else None, 'status_disagreement_upper': disagreement_upper,
                'status_alpha': .05 / 14, 'mean_tv': mean(tv) if tv else None, 'tv_upper': _upper(tv),
                'tv_alpha': .05, 'group_values': values,
                'gate': _gate(registered, [disagreement_upper, _upper(tv)], .01)}
        excess = paired_excess({group['group_id']: group['comparisons'][f'{branch}_to_oracle']['language'] for group in panels['iid']},
            {group['group_id']: group['comparisons']['concat_to_oracle']['language'] for group in panels['iid']}) if branch != 'concat' else None
        absolute = _gate(registered, [iid['comparisons'][f'{branch}_to_oracle']['language_tv_upper']], .02)
        gates = {'absolute': absolute, 'critical': _gate(registered, [value['upper'] for value in critical.values()], .01),
            'noninferiority': _gate(registered, [excess['upper']], .005) if excess else 'not_applicable',
            'paraphrase': 'not_run_registered_dataset' if not registered else ('passed' if all(value['gate'] == 'passed' for value in para.values()) else 'failed'),
            'reproducibility': 'not_run_external_replay_required'}
        gates['numeric_combined'] = ('not_run_registered_dataset' if not registered else
            'passed' if all(gates[key] in ('passed', 'not_applicable') for key in ('absolute', 'critical', 'noninferiority', 'paraphrase')) else 'failed')
        st = structure[branch]
        structural = structural_scores(st['predictions'], st['targets'], st['normalize'])
        structural.pop('critical_errors')
        structural_panels = {}
        for label in ('iid',) + STRATA:
            indices = [i for i, (split, kind, stratum) in enumerate(st['groups'])
                       if ('iid' if split == 'iid_test' else f'{kind}/{stratum}') == label]
            if indices:
                value = structural_scores([st['predictions'][i] for i in indices],
                    [st['targets'][i] for i in indices], [st['normalize'][i] for i in indices])
                value.pop('critical_errors')
                structural_panels[label] = value
        branches[branch] = {'structural': structural, 'structural_panels': structural_panels, 'critical_strata': critical, 'paraphrase_strata': para,
            'paired_excess_to_concat': excess, 'coverage': dict(coverage[branch]) | {'point_fraction': coverage[branch]['point_queries'] / coverage[branch]['queries'],
                'statuses': dict(status_counts[branch])}, 'gates': gates}
    kernel = compare_kernels(core, SIMULATOR)
    oracle = iid['comparisons']['oracle_to_simulator']
    numeric_oracle_gate = _gate(registered, [oracle['language_tv_upper'], oracle['joint_tv_upper']], .02)
    if registered and kernel['mismatches']:
        numeric_oracle_gate = 'failed'
    report = {'schema': 'l1-heldout-evaluation-v1', 'protocol_id': 'L1-v0.1', 'registered_size': registered,
        'core_artifact_id': core.artifact_id, 'audit': audit, 'iid': iid, 'challenge': challenge,
        'branches': branches, 'numeric_oracle_gate': numeric_oracle_gate, 'oracle_validation_gate': 'external_validation_preflight_required',
        'kernel': kernel, 'statistical_unit': 'independent_base_group', 'seed_replication_unit': False,
        'geometry_diagnostic': 'deferred', 'test_feedback_to_training': False,
        'rare_contexts': {'status': 'not_established', 'threshold': .01,
            'reason': 'Protocol specifies no context projection; semantic-hash ownership and rejection condition the generator. Exact conditional probability for a declared projection has not been established.',
            'scores': None, 'training_frequency_is_probability': False},
        'external_evidence_required': ['whole-study configuration/checkpoint freeze before this call', 'G0 analytic/status fixtures and exact TTT',
            'G1 actual train-loader/test isolation', 'exact AnswerRecord replay'],
        'scope': 'deterministic known-graph fixed-SCM conditional laws; no identification certificate inferred',
        'L1_status': 'not_run_external_evidence_required' if registered else 'not_run_registered_dataset'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    return report
