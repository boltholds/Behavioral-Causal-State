"""Executable reference experiments. Run from repository root or pass --repo-root."""
import argparse
from dataclasses import asdict
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import scipy
from . import __version__
from .generator import generate, public_record, semantic_key, TEMPLATES, NAMES
from .inference import History, Evidence, Predictive, Interventional, Counterfactual, ExactDistribution, evaluate
from .simulator import WorldSpec, Regime, Step, Action, Variable as V, Clamp
from .misspecification import Interval, NoiseClass, NotEstablished, EXPECTED, certify, population_panel, run_panel, clopper_pearson
from .population import ResponseModel, response_bounds, SearchArm, search, marginal_conflict
from .contracts import CertifiedBounds, Undefined


def manifest(root,protocol_path):
    sources = sorted((Path(__file__).parent).glob('*.py'))
    return {'runtime_version':__version__,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'protocol_sha256':sha256(protocol_path.read_bytes()).hexdigest(),
            'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in sources},
            'platform':platform.platform(),'command':sys.argv,'rng':'NumPy PCG64; seeded independent streams',
            'protocol_path':str(protocol_path.relative_to(root))}


def _load(root,name):
    path = root / 'experiments/protocols' / name
    return path,json.loads(path.read_text())


def _coupling_control():
    minus,plus = ResponseModel(F(0)),ResponseModel(F(1,4))
    bounds = response_bounds(Interval(F(1,4),F(1,4)),Interval(F(3,4),F(3,4)))
    return minus.marginals() == plus.marginals() and minus.query() == 0 and plus.query() == 1 and bounds == CertifiedBounds(0,1)


def g2_report(root,smoke=False):
    path,protocol = _load(root,'misspecification-v0.1.json')
    settings = protocol['finite_data']
    if settings['bonferroni_cells'] != 12 or settings['simultaneous_panel_alpha'] != .05:
        raise ValueError('this runtime implements only the registered 12-cell alpha=.05 panel')
    low,high = settings['seed_range_inclusive']
    seeds = tuple(range(low,min(high+1,low+2) if smoke else high+1))
    started = perf_counter()
    report = run_panel(seeds,settings['episodes_per_regime_cell_seed'])
    exact_ok = all(report['exact_compatibility'][r][m.value] == expected for r,expected_pair in EXPECTED.items() for m,expected in zip(NoiseClass,expected_pair))
    budget_ok = all(isinstance(certify(tuple(Interval(x,x) for x in population_panel(r)),m,budget=0),NotEstablished) for r in EXPECTED for m in NoiseClass)
    controls = {'exact_matrix':exact_ok,'budget_fault':budget_ok,'undetectable_coupling':_coupling_control()}
    acceptance = protocol['acceptance']
    incompatible = all(report['rejections'][r][m.value] >= acceptance['minimum_rejections_per_incompatible_combination'] for r,wants in EXPECTED.items() for m,want in zip(NoiseClass,wants) if not want)
    compatible = all(report['rejections'][r][m.value] <= acceptance['maximum_rejections_per_compatible_combination'] for r,wants in EXPECTED.items() for m,want in zip(NoiseClass,wants) if want)
    complete = all(not value.startswith('NotEstablished') for row in report['rows'] for value in row['statuses'].values())
    gates = {'exact_matrix':exact_ok,'detection':incompatible,'false_rejection_control':compatible,'budget_fault':budget_ok,'undetectable_coupling':controls['undetectable_coupling'],'certificates_complete':complete}
    return report | {'protocol_id':protocol['protocol_id'],'mode':'smoke' if smoke else 'registered_full','acceptance_status':'not_run_full_protocol' if smoke else 'passed' if all(gates.values()) else 'failed','gates':gates,'controls':controls,'manifest':manifest(root,path),'wall_seconds':perf_counter()-started}


def p1_controls():
    p0,p1 = Interval(F(1,4),F(1,4)),Interval(F(3,4),F(3,4))
    rng = np.random.default_rng(20261004)
    counts = (int(rng.binomial(1000,.25)),int(rng.binomial(1000,.75)))
    controls = {
        'coupling':{'passed':_coupling_control(),'population_bounds':[0,1]},
        'monotonicity':{'passed':response_bounds(p0,p1,True) == CertifiedBounds(1,1),'bounds':asdict(response_bounds(p0,p1,True))},
        'zero_denominator':{'passed':isinstance(response_bounds(Interval(F(0),F(1,4)),p1),Undefined),'policy':'Undefined when factual denominator can vanish'},
        'infeasible':{'passed':marginal_conflict(p0,Interval(F(0),F(0))),'scope':'PopulationLaw','proof':'P(10)+P(11)=1/4 and P(10)+P(11)=0 are disjoint exact equalities'},
        'restricted_coupling':{'passed':ResponseModel(F(1,8)).query() == F(1,2),'scope':'DeclaredFiniteClass','t':'1/8','point':float(ResponseModel(F(1,8)).query()),'does_not_identify_full_class':True},
    }
    try:
        intervals = tuple(clopper_pearson(k,1000,F(1,40)) for k in counts)
        bounds = response_bounds(*intervals)
        contains_truth = intervals[0].lower <= F(1,4) <= intervals[0].upper and intervals[1].lower <= F(3,4) <= intervals[1].upper
        controls['finite_data'] = {'passed':contains_truth and isinstance(bounds,CertifiedBounds) and bounds.lower == 0 and bounds.upper == 1,
            'scope':'FiniteDataConstraints','seed':20261004,'successes':counts,'trials_per_arm':1000,'simultaneous_confidence':.95,'correction':'Bonferroni across two marginals; cell alpha=1/40',
            'sampling_intervals':[[float(i.lower),float(i.upper)] for i in intervals],
            'coupling_bounds':asdict(bounds),'population_coupling_bounds':[0,1]}
    except ArithmeticError as error:
        controls['finite_data'] = {'passed':False,'status':'NotEstablished:NumericalFailure','reason':str(error)}
    return controls


def assess_p1(summary,controls,minimum,total,evaluations):
    required = {'coupling','monotonicity','finite_data','zero_denominator','infeasible','restricted_coupling'}
    return (set(summary) == {a.value for a in SearchArm}
            and set(controls) == required
            and all(c.get('passed') is True for c in controls.values())
            and all(s['successful_seeds'] >= minimum and s['total_seeds'] == total and s['evaluations_per_run'] == evaluations for s in summary.values()))


def p1_report(root,smoke=False):
    path,protocol = _load(root,'population-v0.1.json')
    settings = protocol['search']
    if settings['population_size'] != 64 or settings['offspring_per_generation'] != 64 or settings['initial_uniform_interval'] != [.1,.15] or settings['boundary_operator'] != 'clip':
        raise ValueError('unsupported P1 protocol settings')
    low,high = settings['seed_range_inclusive']
    seeds = tuple(range(low,min(high+1,low+2) if smoke else high+1))
    generations = 3 if smoke else settings['generations']
    rows,summary = [],{}
    for arm in SearchArm:
        for seed in seeds:
            result = search(arm,seed,generations,settings['population_size'],settings['mutation_sigma'])
            row = asdict(result)
            row['scope'] = result.scope.value
            row['identification'] = result.identification.value
            row['arm'] = result.arm.value
            row['result']['kind'] = 'WitnessRange'
            rows.append(row)
        selected = [row for row in rows if row['arm'] == arm.value]
        successes = sum(row['result']['lower'] <= protocol['gates']['found_min_at_most'] and row['result']['upper'] >= protocol['gates']['found_max_at_least'] for row in selected)
        summary[arm.value] = {'successful_seeds':successes,'total_seeds':len(seeds),'evaluations_per_run':64+64*generations,'total_wall_seconds':sum(row['wall_seconds'] for row in selected)}
    controls = p1_controls()
    passed = assess_p1(summary,controls,protocol['gates']['minimum_successful_runs'],protocol['gates']['total_runs'],settings['evaluations_per_run'])

    return {'protocol_id':protocol['protocol_id'],'mode':'smoke' if smoke else 'registered_full','acceptance_status':'not_run_full_protocol' if smoke else 'passed' if passed else 'failed','summary':summary,'rows':rows,'controls':controls,'certified_population_bounds':[0,1],'certificate':'Frechet response-type constraints: 0<=t<=1/4 and q=4t; endpoints supplied by M-/M+','manifest':manifest(root,path)}


def _distribution(result):
    if not isinstance(result,ExactDistribution):
        return {'status':type(result).__name__,'reason':result.reason}
    return {'probabilities':[{'outcome':list(x),'probability':str(p)} for x,p in result.probabilities], 'evidence_probability':str(result.evidence_probability),'transitions':result.transitions}


def demo():
    spec = WorldSpec.for_regime(Regime.STOCHASTIC_PARTIAL)
    history = History((Step.command(0,Action.START),),(Evidence(0,0,V.C,0),Evidence(1,0,V.Y,1)))
    return {'scope':'DeclaredFiniteClass','assumptions':['known XOR mechanisms','independent Bernoulli(1/10) mechanism noise','iid across time and objects','fixed action schedule'],
            'history':'C_0=0; executed start at t=1; observed Y_1=1',
            'counterfactual':_distribution(evaluate(spec,history,Counterfactual(0,1,Action.STOP,(V.Y,)))),
            'fresh_intervention':_distribution(evaluate(spec,history,Interventional(0,(Clamp(0,V.R,0),),(V.Y,)))),
            'interpretation':'CF replaces past start using abducted shared U; fresh do integrates a new time slice.'}


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    subs = parser.add_subparsers(dest='command',required=True)
    subs.add_parser('demo')
    for command in ('g2','p1'):
        p = subs.add_parser(command)
        p.add_argument('--smoke',action='store_true')
        p.add_argument('--output',type=Path,required=True)
    p = subs.add_parser('generate')
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--regime',choices=[r.value for r in Regime],default='deterministic')
    p.add_argument('--train-labels',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            print(json.dumps(demo(),ensure_ascii=False,indent=2))
            return 0
        if args.command in ('g2','p1'):
            report = (g2_report if args.command == 'g2' else p1_report)(args.repo_root.resolve(),args.smoke)
            _write(args.output,report)
            print(json.dumps({'output':str(args.output),'acceptance_status':report['acceptance_status']}))
            return 1 if report['acceptance_status'] == 'failed' else 0
        if args.count <= 0 or args.seed < 0:
            raise ValueError('positive count and nonnegative seed required')
        spec = WorldSpec.for_regime(Regime(args.regime))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        keys = []
        with args.output.open('w') as file:
            for i in range(args.count):
                history = generate(spec,args.seed,f'base-{i}')
                file.write(json.dumps(public_record(history,args.train_labels),ensure_ascii=False)+'\n')
                keys.append(semantic_key(history.events))
        _write(args.output.with_suffix('.manifest.json'),{'kind':'base_generation_only','seed':args.seed,'regime':args.regime,'count':args.count,'unique_semantic_histories':len(set(keys)),'templates':TEMPLATES,'names':NAMES,'G1':'not_run_full_challenge_dataset','labels_exported':args.train_labels,'output_sha256':sha256(args.output.read_bytes()).hexdigest()})
        print(json.dumps({'output':str(args.output),'count':args.count}))
        return 0
    except (ValueError,ArithmeticError,OSError) as error:
        parser.exit(2,f'error: {error}\n')
