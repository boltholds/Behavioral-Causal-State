"""Train-only decoder memorization diagnostic; never an L1 acceptance run."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import zip_longest
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch

from bcs.embedding_cache import load_cache
from bcs.grounder import Grounder, ReaderKind
from bcs.language_evaluation import structural_report
from bcs.language_lock import source_hashes, verify_input_lock
from bcs.language_tokens import BOS, EOS, FIELDS, PAD
from bcs.language_training import load_checkpoint, load_labels
from bcs.sonar_encoder import file_hash

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'experiments/protocols/decoder-memorization-v0.1.json'


def select_indices(lengths, targets):
    if len(lengths) != len(targets):
        raise ValueError('length/target alignment mismatch')
    selected = []
    for length in (4, 8, 12, 16):
        seen = set(); current = []
        for i, (n, target) in enumerate(zip(lengths, targets)):
            if n == length and tuple(target) not in seen:
                seen.add(tuple(target)); current.append(i)
            if len(current) == 2:
                break
        if len(current) != 2:
            raise ValueError(f'two distinct target histories of length {length} required')
        selected.extend(current)
    return tuple(selected)


def same_length_permutation(lengths, targets):
    if len(lengths) != len(targets) or not len(lengths):
        raise ValueError('nonempty aligned control batch required')
    permutation = list(range(len(lengths)))
    for length in set(lengths):
        group = [i for i, n in enumerate(lengths) if n == length]
        if len(group) != 2 or tuple(targets[group[0]]) == tuple(targets[group[1]]):
            raise ValueError('each length needs exactly two distinct target histories')
        a, b = group; permutation[a], permutation[b] = b, a
    return tuple(permutation)


def sequence_diagnostics(predictions, targets):
    report = structural_report(predictions, targets)
    first = Counter(); termination = 0
    for predicted, target in zip(predictions, targets):
        p, t = tuple(predicted), tuple(target)
        termination += p.count(EOS) != 1 or p.index(EOS) != len(t)-1 or len(p) != len(t)
        for i, (actual, expected) in enumerate(zip_longest(p, t, fillvalue=-1)):
            if actual != expected:
                first['termination' if EOS in (actual, expected) or -1 in (actual, expected) else FIELDS[i % 6]] += 1
                break
    report.update(termination_errors=termination, first_error_fields=dict(first))
    return report


def _targets_tensor(targets, device):
    if not targets or any(not target for target in targets):
        raise ValueError('nonempty target sequences required')
    result = torch.full((len(targets), max(map(len, targets))), PAD, dtype=torch.long, device=device)
    for i, target in enumerate(targets):
        result[i, :len(target)] = torch.tensor(target, device=device)
    return result


@torch.inference_mode()
def teacher_step_parity(model, z, lengths, targets):
    """Compare vectorized teacher forcing with incremental identical gold prefixes."""
    model.eval(); target = _targets_tensor(targets, z.device)
    vectorized = model.teacher_logits(z, lengths, target)
    context, memory, padding = model._read(z, lengths)
    hidden = context[None, :, :]
    previous = torch.full((len(z), 1), BOS, dtype=torch.long, device=z.device)
    steps = []
    for i in range(target.shape[1]):
        x = torch.cat((model.tokens(previous), context[:, None, :]), dim=-1)
        decoded, hidden = model.decoder(x, hidden)
        steps.append(model._logits(decoded, memory, padding))
        previous = target[:, i:i+1]
    incremental = torch.cat(steps, dim=1)
    valid = vectorized != torch.finfo(vectorized.dtype).min
    delta = (vectorized[valid] - incremental[valid]).abs()
    incremental = incremental.masked_fill(~valid, torch.finfo(incremental.dtype).min)
    disagreements = ((vectorized.argmax(-1) != incremental.argmax(-1)) & (target != PAD)).sum()
    return {'max_abs_valid_logit_difference': float(delta.max()),
            'argmax_disagreements': int(disagreements),
            'valid_logits_allclose': bool(torch.allclose(vectorized[valid], incremental[valid], atol=1e-4, rtol=1e-5))}


@torch.inference_mode()
def measure(model, z, lengths, targets):
    model.eval()
    predictions = model.greedy(z, lengths)  # Targets never enter the free-running path.
    target = _targets_tensor(targets, z.device)
    logits = model.teacher_logits(z, lengths, target)
    mask = target != PAD; wrong = (logits.argmax(-1) != target) & mask
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD, reduction='sum')
    fields = {name: {'errors': 0, 'tokens': 0} for name in (*FIELDS, 'termination')}
    for row, tokens in enumerate(targets):
        for i, token in enumerate(tokens):
            field = 'termination' if token == EOS else FIELDS[i % 6]
            fields[field]['tokens'] += 1
            fields[field]['errors'] += int(wrong[row, i])
    return {'free': sequence_diagnostics(predictions, targets),
            'teacher': {'nll': float(loss) / int(mask.sum()), 'tokens': int(mask.sum()),
                        'token_errors': int(wrong.sum()), 'token_error': float(wrong.sum() / mask.sum()),
                        'full_history_error': float(wrong.any(1).float().mean()), 'fields': fields}}


@dataclass(frozen=True)
class FitConfig:
    max_steps: int = 1000
    evaluate_every: int = 25
    required_successes: int = 3
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    dropout: float = 0.1
    seed: int = 11

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_steps, self.evaluate_every, self.required_successes)) or type(self.seed) is not int or self.seed < 0:
            raise ValueError('positive step/check counts and nonnegative integer seed required')
        if not all(math.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.gradient_clip_norm, self.dropout)) or self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0 or not 0 <= self.dropout < 1:
            raise ValueError('invalid optimizer/dropout configuration')


def validate_protocol(protocol):
    fixed = {
        'protocol_id': 'decoder-memorization-v0.1', 'scope': 'train_set_memorization_diagnostic',
        'L1_status': 'not_run', 'branches': ['concat', 'attention', 'gru'],
        'history_lengths': [4, 8, 12, 16], 'histories_per_length': 2,
        'selection': 'first_distinct_target_histories_in_locked_train_order',
        'single_history': 'first_selected_length_4', 'batching': 'full_selected_batch', 'device': 'cpu',
        'success': 'all_exact_greedy_histories_and_zero_invalid_for_3_consecutive_checks',
        'controls': ['zero_embeddings_same_lengths', 'swap_each_same_length_pair'],
        'generalization_evaluation': 'not_run', 'architecture_changes': 'none', 'normalization': 'none',
    }
    if set(protocol) != set(fixed) | {'training', 'threads'} or any(protocol[k] != v for k, v in fixed.items()):
        raise ValueError('unsupported fixed diagnostic protocol declaration')
    config = FitConfig(**protocol['training'])
    if config.required_successes != 3 or type(protocol['threads']) is not int or protocol['threads'] < 1:
        raise ValueError('invalid diagnostic protocol success/threads declaration')
    return config


@dataclass
class SuccessStreak:
    required: int
    count: int = 0

    def __post_init__(self):
        if type(self.required) is not int or self.required < 1:
            raise ValueError('positive success streak required')

    def observe(self, exact):
        if type(exact) is not bool:
            raise ValueError('boolean success required')
        self.count = self.count + 1 if exact else 0
        return self.count >= self.required


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def _implementation_identity():
    return {'runtime_source_sha256': source_hashes(), 'diagnostic_source_sha256': file_hash(__file__),
            'protocol_sha256': file_hash(PROTOCOL)}


def restore_fit(output, expected_provenance):
    output = Path(output)
    metadata = json.loads((output/'checkpoint.json').read_text())
    report = json.loads((output/'report.json').read_text())
    if metadata['schema'] != 'memorization-checkpoint-v1' or metadata['scope'] != 'train_set_memorization_diagnostic' or metadata['provenance'] != expected_provenance or metadata['weights_sha256'] != file_hash(output/'model.pt') or report['checkpoint_sha256'] != file_hash(output/'checkpoint.json'):
        raise ValueError('memorization checkpoint provenance/checksum mismatch')
    if metadata.get('implementation') != _implementation_identity():
        raise ValueError('memorization checkpoint implementation mismatch')
    model = Grounder(ReaderKind(metadata['reader']), metadata['input_dim'], metadata['config']['dropout'])
    model.load_state_dict(torch.load(output/'model.pt', map_location='cpu', weights_only=True))
    return model.eval()


def fit(kind, z, lengths, targets, output, config, provenance, progress=lambda row: None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('fit destination must be empty')
    if z.device.type != 'cpu' or len(z) != len(targets):
        raise ValueError('aligned CPU diagnostic batch required')
    implementation = _implementation_identity()
    permutation = same_length_permutation(lengths.tolist(), targets) if len(z) > 1 else ()
    torch.manual_seed(config.seed); torch.use_deterministic_algorithms(True)
    model = Grounder(kind, z.shape[2], config.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    target = _targets_tensor(targets, z.device)
    started = perf_counter(); history = []
    parity = teacher_step_parity(model, z, lengths, targets)
    if not parity['valid_logits_allclose'] or parity['argmax_disagreements']:
        raise ValueError('teacher/incremental implementation mismatch')
    history.append({'step': 0, **measure(model, z, lengths, targets)})
    gate = SuccessStreak(config.required_successes); status = 'budget_exhausted'
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, config.max_steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model.teacher_logits(z, lengths, target)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite memorization loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm); optimizer.step()
        if step % config.evaluate_every == 0 or step == config.max_steps:
            measured = measure(model, z, lengths, targets)
            row = {'step': step, 'update_nll': float(loss.detach()), **measured}
            history.append(row)
            exact = measured['free']['full_history_error'] == 0 and measured['free']['invalid_decodings'] == 0
            success = gate.observe(exact)
            progress({'reader': kind.value, 'histories': len(z), 'step': step,
                      'teacher_nll': measured['teacher']['nll'],
                      'full_history_error': measured['free']['full_history_error'], 'success_streak': gate.count})
            if success:
                status = 'memorized'; break
    model.eval()
    original = model.greedy(z, lengths)
    zero = model.greedy(torch.zeros_like(z), lengths)
    controls = {'zero_embeddings': sequence_diagnostics(zero, targets)}
    predictions = {'targets': targets, 'original': original, 'zero_embeddings': zero}
    if permutation:
        swapped = model.greedy(z[list(permutation)], lengths)
        donor = tuple(targets[i] for i in permutation)
        controls['same_length_swap'] = {'status': 'evaluated', 'permutation': permutation,
                                        'recipient_targets': sequence_diagnostics(swapped, targets),
                                        'donor_targets': sequence_diagnostics(swapped, donor)}
        predictions['same_length_swap'] = swapped
    else:
        controls['same_length_swap'] = {'status': 'not_applicable_single_history'}
    final_parity = teacher_step_parity(model, z, lengths, targets)
    if not final_parity['valid_logits_allclose'] or final_parity['argmax_disagreements']:
        raise ValueError('trained teacher/incremental implementation mismatch')
    torch.save(model.state_dict(), output/'model.pt')
    checkpoint = {'schema': 'memorization-checkpoint-v1', 'scope': 'train_set_memorization_diagnostic',
                  'reader': kind.value, 'input_dim': model.input_dim, 'config': asdict(config),
                  'provenance': provenance, 'implementation': implementation,
                  'weights_sha256': file_hash(output/'model.pt')}
    _write(output/'checkpoint.json', checkpoint)
    report = {'schema': 'memorization-fit-v1', 'scope': 'train_set_memorization_diagnostic',
              'L1_status': 'not_run', 'generalization_evaluation': 'not_run', 'reader': kind.value,
              'status': status, 'histories': len(z), 'config': asdict(config), 'optimizer_steps': step,
              'success_streak': gate.count, 'history': history, 'final': history[-1], 'controls': controls,
              'teacher_step_parity_initial': parity, 'teacher_step_parity_final': final_parity,
              'parameters': sum(p.numel() for p in model.parameters()), 'wall_seconds': perf_counter()-started,
              'checkpoint_sha256': file_hash(output/'checkpoint.json'), 'provenance': provenance}
    _write(output/'predictions.json', predictions); _write(output/'report.json', report)
    restored = restore_fit(output, provenance)
    if restored.greedy(z, lengths) != original:
        raise ValueError('restored predictions differ')
    report['restoration_verified'] = True
    _write(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'cache-root', 'baseline-runs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        protocol = json.loads(PROTOCOL.read_text())
        config = validate_protocol(protocol)
        input_lock = verify_input_lock(args.dataset, args.cache_root, ROOT/'experiments/protocols/language-v0.1.json')
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('experiment destination must be empty')
        torch.set_num_threads(protocol['threads'])
        cache = load_cache(args.cache_root/'train', args.dataset/'public/train.jsonl')
        targets = load_labels(args.dataset/'training/labels.jsonl', cache)
        indices = select_indices(tuple(map(len, cache.indices)), targets)
        selection = {stage: [{'index': i, 'record_id': cache.record_ids[i], 'length': len(cache.indices[i])} for i in rows]
                     for stage, rows in (('single', indices[:1]), ('eight', indices))}
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output/'selection.json', selection)
        provenance = {'input_lock_sha256': input_lock, 'protocol_sha256': file_hash(PROTOCOL),
                      'diagnostic_source_sha256': file_hash(__file__), 'selection_sha256': file_hash(args.output/'selection.json')}
        _write(args.output/'provenance.json', provenance)
        z8, n8 = cache.batch(indices); t8 = tuple(targets[i] for i in indices)
        z8, n8 = torch.from_numpy(z8), torch.from_numpy(n8)
        baseline = {}
        for branch in protocol['branches']:
            model = load_checkpoint(args.baseline_runs/branch, cache.manifest['encoder'], input_lock)
            previous = json.loads((args.baseline_runs/branch/'report.json').read_text())
            baseline[branch] = {'optimizer_steps': previous['optimizer_steps'], 'best_epoch': previous['best_epoch'],
                                'weights_sha256': file_hash(args.baseline_runs/branch/'model.pt'),
                                'selected_train_metrics': measure(model, z8, n8, t8),
                                'teacher_step_parity': teacher_step_parity(model, z8, n8, t8)}
        _write(args.output/'baseline.json', baseline)
        results = {}
        for stage, rows in (('single', indices[:1]), ('eight', indices)):
            z, n = cache.batch(rows); selected_targets = tuple(targets[i] for i in rows)
            for branch in protocol['branches']:
                key = f'{stage}/{branch}'
                def progress(row):
                    print(json.dumps({'stage': stage, **row}), file=sys.stderr, flush=True)
                result = fit(ReaderKind(branch), torch.from_numpy(z), torch.from_numpy(n), selected_targets,
                             args.output/key, config, {**provenance, 'stage': stage}, progress)
                results[key] = {k: result[k] for k in ('status', 'optimizer_steps', 'success_streak', 'wall_seconds', 'restoration_verified')}
                results[key].update(final=result['final'], controls=result['controls'])
        if verify_input_lock(args.dataset, args.cache_root, ROOT/'experiments/protocols/language-v0.1.json') != input_lock or file_hash(PROTOCOL) != provenance['protocol_sha256'] or file_hash(__file__) != provenance['diagnostic_source_sha256'] or file_hash(args.output/'selection.json') != provenance['selection_sha256']:
            raise ValueError('diagnostic inputs changed during run')
        summary = {'schema': 'memorization-summary-v1', 'scope': protocol['scope'], 'L1_status': 'not_run',
                   'generalization_evaluation': 'not_run', 'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
                   'threads': protocol['threads'], 'device': 'cpu', 'config': asdict(config),
                   'provenance': provenance, 'results': results, 'input_lock_reverified': True}
        _write(args.output/'summary.json', summary)
        print(json.dumps({'output': str(args.output), 'L1_status': 'not_run', 'results': {k: v['status'] for k, v in results.items()}}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
