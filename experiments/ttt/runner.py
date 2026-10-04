"""Reset/MQ simulator adapter and independent exhaustive finite Mealy checks."""
from collections import deque
from pathlib import Path
import json
from bcs.simulator import Action, Clamp, Regime, Step, Variable, WorldSpec

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'experiments/ttt/fixtures-v0.1.json'


def load_fixtures():
    data = json.loads(FIXTURES.read_text())
    return [dict(f, alphabet=data['profiles'][f['profile']]) for f in data['fixtures']]


def decode_step(symbol):
    return Step(tuple(Action(a) for a in symbol['actions']),
                tuple(Clamp(o, Variable(v), b) for o, v, b in symbol['clamps']))


def encode_output(states):
    # Injective full after-step observation: R0,M0,C0,Y0,R1,M1,C1,Y1, MSB first.
    value = 0
    for state in states:
        for variable in Variable:
            value = 2 * value + state.get(variable)
    return value


class SimulatorOracle:
    """Learner sees only reset-based output words, never DeviceState/reference IDs."""
    def __init__(self, fixture, regime=Regime.DETERMINISTIC):
        if regime != Regime.DETERMINISTIC:
            raise ValueError('TTT requires the deterministic regime')
        self.spec = WorldSpec.for_regime(regime)
        self.initial = tuple(self.spec.reset(c, 0, 0) for c in fixture['initial_c'])
        self.alphabet = tuple(decode_step(x) for x in fixture['alphabet'])
        self.mq_count = self.reset_count = self.symbols_executed = 0

    def advance(self, state, symbol):
        step = self.alphabet[symbol]
        return tuple(self.spec.advance(state[o], step.actions[o], step.local_clamps(o), 0, 0)
                     for o in range(2))

    def membership(self, word, prefix_length=0):
        if type(prefix_length) is not int or not 0 <= prefix_length <= len(word):
            raise ValueError('invalid prefix length')
        if any(type(i) is not int or not 0 <= i < len(self.alphabet) for i in word):
            raise ValueError('input symbol outside frozen alphabet')
        self.mq_count += 1
        self.reset_count += 1
        state, output = self.initial, []
        for symbol in word:
            state = self.advance(state, symbol)
            output.append(encode_output(state))
            self.symbols_executed += 1
        return output[prefix_length:]


def build_reference(fixture):
    """Evaluator-only BFS of every full physical state reachable from fixed reset."""
    oracle = SimulatorOracle(fixture)
    states, indices, transitions = [oracle.initial], {oracle.initial: 0}, []
    for state in states:
        row = []
        for symbol in range(len(oracle.alphabet)):
            successor = oracle.advance(state, symbol)
            if successor not in indices:
                indices[successor] = len(states)
                states.append(successor)
            row.append([indices[successor], encode_output(successor)])
        transitions.append(row)
    return {'initial': 0, 'transitions': transitions}


def minimal_state_count(machine):
    """Independent Moore partition refinement for Mealy future-output equivalence."""
    rows = machine['transitions']
    partition = [0] * len(rows)
    while True:
        signatures = [tuple((out, partition[dst]) for dst, out in row) for row in rows]
        classes = {}
        refined = [classes.setdefault(signature, len(classes)) for signature in signatures]
        if refined == partition:
            return len(classes)
        partition = refined


def validate_machine(machine, alphabet_size=None):
    """Validate the complete wire table before any Python indexing can alias states."""
    if not isinstance(machine, dict) or set(machine) != {'initial', 'transitions'}:
        raise ValueError('invalid machine fields')
    rows = machine['transitions']
    if not isinstance(rows, list) or not rows:
        raise ValueError('machine states must be a nonempty list')
    initial = machine['initial']
    if type(initial) is not int or not 0 <= initial < len(rows):
        raise ValueError('invalid initial state index')
    if alphabet_size is None:
        if not isinstance(rows[0], list):
            raise ValueError('invalid machine transition row')
        alphabet_size = len(rows[0])
    if type(alphabet_size) is not int or alphabet_size <= 0:
        raise ValueError('invalid machine alphabet size')
    for row in rows:
        if not isinstance(row, list) or len(row) != alphabet_size:
            raise ValueError('machine alphabet/row mismatch')
        for transition in row:
            if not isinstance(transition, list) or len(transition) != 2:
                raise ValueError('invalid machine transition shape')
            destination, output = transition
            if type(destination) is not int or not 0 <= destination < len(rows):
                raise ValueError('invalid destination state index')
            if type(output) is not int or not 0 <= output <= 255:
                raise ValueError('invalid full-observation output')
    return alphabet_size


def product_counterexample(reference, hypothesis, stats=None):
    """Shortest distinguishing input word, or None after complete product BFS."""
    alphabet_size = validate_machine(reference)
    validate_machine(hypothesis, alphabet_size)
    if stats is None:
        stats = {}
    stats.update(pairs_visited=0, edges_compared=0)
    start = (reference['initial'], hypothesis['initial'])
    queue, seen = deque([(start, [])]), {start}
    while queue:
        (r, h), word = queue.popleft()
        stats['pairs_visited'] += 1
        rr, hr = reference['transitions'][r], hypothesis['transitions'][h]
        if len(rr) != len(hr):
            raise ValueError('hypothesis alphabet mismatch')
        for symbol, ((rn, ro), (hn, ho)) in enumerate(zip(rr, hr)):
            stats['edges_compared'] += 1
            extended = word + [symbol]
            if ro != ho:
                return extended
            pair = (rn, hn)
            if pair not in seen:
                seen.add(pair)
                queue.append((pair, extended))
    return None


def digest(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def prepare_java(cache):
    """Download only pinned official bundle bytes and compile via Java 17's compiler."""
    import subprocess
    import urllib.request
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / 'experiments/ttt/dependencies-v0.1.json').read_text())
    jars = []
    for dependency in lock['dependencies']:
        path = cache / dependency['filename']
        if not path.exists():
            data = urllib.request.urlopen(dependency['url'], timeout=90).read()
            from hashlib import sha256
            if sha256(data).hexdigest() != dependency['sha256']:
                raise ValueError('downloaded dependency checksum mismatch')
            path.write_bytes(data)
        if digest(path) != dependency['sha256'] or path.stat().st_size != dependency['bytes']:
            raise ValueError('dependency checksum mismatch')
        jars.append(path)
    java_version = subprocess.run(['java', '-version'], capture_output=True, text=True, check=True).stderr
    if not java_version.startswith('openjdk version "17.'):
        raise ValueError('Java 17 required by the frozen protocol')
    source = ROOT / 'experiments/ttt/LearnLibTTT.java'
    classes = cache / ('ttt-classes-' + digest(source)[:16])
    classes.mkdir(exist_ok=True)
    classpath = ':'.join(map(str, jars))
    subprocess.run(['java', '-XX:ActiveProcessorCount=2', '-m', 'jdk.compiler/com.sun.tools.javac.Main',
                    '--release', '17', '-cp', classpath, '-d', str(classes), str(source)],
                   check=True, capture_output=True, text=True)
    import zipfile
    from hashlib import sha256
    with zipfile.ZipFile(jars[0]) as jar:
        ttt_hash = sha256(jar.read('de/learnlib/algorithm/ttt/mealy/TTTLearnerMealy.class')).hexdigest()
    return {'classpath': str(classes) + ':' + classpath, 'java_version': java_version.strip(),
            'compiled_classes_sha256': {p.name: digest(p) for p in sorted(classes.glob('LearnLibTTT*.class'))},
            'learnlib_ttt_class_sha256': ttt_hash,
            'class_sha256': digest(classes / 'LearnLibTTT.class'),
            'dependencies': lock['dependencies']}


def machine_output(machine, word):
    state, out = machine['initial'], []
    for symbol in word:
        state, output = machine['transitions'][state][symbol]
        out.append(output)
    return out


def run_trial(fixture, repetition, build):
    """Fresh Java learner; EQ receives its full exported model, never passes states back."""
    import subprocess
    import time
    oracle, reference = SimulatorOracle(fixture), build_reference(fixture)
    started = time.monotonic()
    process = subprocess.Popen(['java', '-ea', '-XX:ActiveProcessorCount=2', '-Xmx256m',
                                '-cp', build['classpath'], 'LearnLibTTT', str(len(fixture['alphabet']))],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    eq_count, counterexamples, eq_checks, membership_queries = 0, [], [], []
    try:
        for line in process.stdout:
            kind, payload = line.rstrip('\n').split(' ', 1)
            if kind == 'MQ':
                parts = list(map(int, payload.split()))
                output = oracle.membership(parts[1:], parts[0])
                membership_queries.append([parts[0], parts[1:], output])
                reply = ' '.join(map(str, output)) + '\n'
            elif kind == 'EQ':
                eq_count += 1
                hypothesis = json.loads(payload)
                eq_stats = {}
                ce = product_counterexample(reference, hypothesis, stats=eq_stats)
                eq_checks.append(dict(eq_stats, counterexample=ce, hypothesis=hypothesis))
                if ce is None:
                    reply = 'OK\n'
                else:
                    counterexamples.append(ce)
                    reply = 'CE ' + ' '.join(map(str, ce)) + '\n' + ' '.join(map(str, machine_output(reference, ce))) + '\n'
            elif kind == 'DONE':
                result = json.loads(payload)
                break
            else:
                raise ValueError('unknown learner protocol message')
            process.stdin.write(reply)
            process.stdin.flush()
        else:
            raise RuntimeError('LearnLib process ended without a result: ' + process.stderr.read())
        process.stdin.close()
        returncode = process.wait(timeout=15)
        stderr = process.stderr.read()
        if returncode:
            raise RuntimeError(f'LearnLib exited {returncode}: {stderr}')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    hypothesis = result['hypothesis']
    remaining = product_counterexample(reference, hypothesis)
    minimal = minimal_state_count(reference)
    passed = remaining is None and len(hypothesis['transitions']) == minimal
    return dict(result, fixture_id=fixture['id'], repetition=repetition,
                status='passed' if passed else 'failed',
                reference_states=len(reference['transitions']), minimal_reference_states=minimal,
                hypothesis_states=len(hypothesis['transitions']), alphabet_size=len(fixture['alphabet']),
                mq_count=oracle.mq_count, reset_count=oracle.reset_count,
                membership_queries=membership_queries,
                symbols_executed=oracle.symbols_executed, eq_count=eq_count,
                counterexamples=counterexamples, eq_checks=eq_checks, remaining_counterexample=remaining,
                elapsed_seconds=round(time.monotonic()-started, 6), java_stderr=stderr.strip())


def source_hashes():
    names = ['src/bcs/simulator.py', 'experiments/ttt/runner.py',
             'experiments/ttt/LearnLibTTT.java', 'experiments/ttt/fixtures-v0.1.json',
             'experiments/ttt/dependencies-v0.1.json', 'experiments/protocols/ttt-v0.1.json']
    return {name: digest(ROOT / name) for name in names}


def object_digest(value):
    from hashlib import sha256
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def derived_summary(trials, fixture_count):
    return {'status': 'passed' if all(t['status'] == 'passed' for t in trials) else 'failed',
            'fixture_count': fixture_count, 'completed_trials': len(trials),
            'total_mq': sum(t['mq_count'] for t in trials),
            'total_eq': sum(t['eq_count'] for t in trials),
            'total_symbols_executed': sum(t['symbols_executed'] for t in trials),
            'remaining_counterexamples': sum(t['remaining_counterexample'] is not None for t in trials)}


def negative_controls(reference, hypothesis):
    import copy
    corrupted = copy.deepcopy(hypothesis)
    corrupted['transitions'][corrupted['initial']][0][1] ^= 1
    trivial = {'initial': 0, 'transitions': [[[0, 0]] * len(reference['transitions'][0])]}
    return {'corrupted_hypothesis_counterexample': product_counterexample(reference, corrupted),
            'trivial_false_conjecture_counterexample': product_counterexample(reference, trivial)}


def validate_report(report):
    """Recompute evidence, including actual MQ transcripts and every EQ conjecture."""
    if report['manifest']['source_sha256'] != source_hashes():
        raise ValueError('source bytes differ from registered run')
    protocol = json.loads((ROOT / 'experiments/protocols/ttt-v0.1.json').read_text())
    if report['protocol'] != protocol['version']:
        raise ValueError('report protocol mismatch')
    lock = json.loads((ROOT / 'experiments/ttt/dependencies-v0.1.json').read_text())
    if report['manifest']['dependencies'] != lock['dependencies']:
        raise ValueError('dependency bytes differ from pinned lock')
    # Fresh compilation prevents trusting a supplied binary identity or cached class.
    build = prepare_java(Path('/tmp/artifacts'))
    for manifest_key, build_key in [('compiled_class_sha256', 'class_sha256'),
            ('compiled_classes_sha256', 'compiled_classes_sha256'),
            ('learnlib_ttt_class_sha256', 'learnlib_ttt_class_sha256')]:
        if report['manifest'].get(manifest_key) != build[build_key]:
            raise ValueError('compiled/upstream class provenance mismatch')
    fixtures = {f['id']: f for f in load_fixtures()}
    expected = {(fid, repetition) for fid in fixtures for repetition in protocol['repetitions']}
    actual = [(t['fixture_id'], t['repetition']) for t in report['trials']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('registered trial coverage mismatch')
    for trial in report['trials']:
        fixture = fixtures[trial['fixture_id']]
        if trial['learner_class'] != protocol['learner']:
            raise ValueError('unregistered learner class')
        hypothesis = trial['hypothesis']
        validate_machine(hypothesis, len(fixture['alphabet']))
        if object_digest(hypothesis) != trial['hypothesis_sha256']:
            raise ValueError('hypothesis bytes differ from registered export')
        ref = build_reference(fixture)
        if object_digest(ref) != trial['reference_sha256']:
            raise ValueError('reference bytes differ from registered run')
        if product_counterexample(ref, hypothesis) is not None:
            raise ValueError('exported hypothesis has an exact counterexample')
        for key in ('mq_count', 'reset_count', 'symbols_executed', 'eq_count'):
            if type(trial[key]) is not int or trial[key] <= 0:
                raise ValueError('invalid query counters')
        queries = trial.get('membership_queries')
        if not isinstance(queries, list) or len(queries) != trial['mq_count']:
            raise ValueError('membership transcript/counter mismatch')
        oracle = SimulatorOracle(fixture)
        for query in queries:
            if not isinstance(query, list) or len(query) != 3:
                raise ValueError('invalid membership transcript shape')
            prefix, word, output = query
            if not isinstance(word, list) or not isinstance(output, list):
                raise ValueError('invalid membership transcript word')
            if any(type(x) is not int or not 0 <= x <= 255 for x in output):
                raise ValueError('invalid membership transcript output')
            if oracle.membership(word, prefix) != output:
                raise ValueError('membership transcript disagrees with simulator')
        if (oracle.mq_count, oracle.reset_count, oracle.symbols_executed) != (
                trial['mq_count'], trial['reset_count'], trial['symbols_executed']):
            raise ValueError('query counters disagree with executed transcript')
        checks = trial.get('eq_checks')
        if not isinstance(checks, list) or len(checks) != trial['eq_count']:
            raise ValueError('equivalence transcript/counter mismatch')
        witnesses = []
        for index, check in enumerate(checks):
            if not isinstance(check, dict) or 'hypothesis' not in check:
                raise ValueError('missing equivalence conjecture evidence')
            stats = {}
            ce = product_counterexample(ref, check['hypothesis'], stats=stats)
            if (ce != check['counterexample'] or stats['pairs_visited'] != check['pairs_visited']
                    or stats['edges_compared'] != check['edges_compared']):
                raise ValueError('equivalence transcript disagrees with full product BFS')
            if index == len(checks) - 1:
                if ce is not None or check['hypothesis'] != hypothesis:
                    raise ValueError('final equivalence conjecture is not the accepted export')
            else:
                if ce is None:
                    raise ValueError('accepted conjecture was spuriously followed by refinement')
                witnesses.append(ce)
        if witnesses != trial['counterexamples']:
            raise ValueError('counterexample history differs from equivalence transcript')
        minimal = minimal_state_count(ref)
        if (trial['status'] != 'passed' or trial['remaining_counterexample'] is not None
                or type(trial['refinements']) is not int or trial['refinements'] != len(witnesses)
                or trial['hypothesis_states'] != len(hypothesis['transitions'])
                or trial['minimal_reference_states'] != minimal
                or trial['hypothesis_states'] != minimal
                or trial['reference_states'] != len(ref['transitions'])
                or trial['alphabet_size'] != len(fixture['alphabet'])):
            raise ValueError('trial metrics or status inconsistent')
    if report['summary'] != derived_summary(report['trials'], len(fixtures)):
        raise ValueError('summary differs from validated trial evidence')
    first = report['trials'][0]
    negatives = negative_controls(build_reference(fixtures[first['fixture_id']]), first['hypothesis'])
    if report['negative_checks'] != negatives or any(ce is None for ce in negatives.values()):
        raise ValueError('negative controls differ from exact checks')
    return {'status': 'passed', 'trials': len(actual), 'remaining_counterexamples': 0}


def run_all(output, cache):
    """Register hashes before opening the fixed 60-trial run, then execute its full budget."""
    import platform
    import sys
    import zipfile
    from datetime import datetime, timezone
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    build = prepare_java(cache)
    fixture_data = json.loads(FIXTURES.read_text())
    protocol = json.loads((ROOT / 'experiments/protocols/ttt-v0.1.json').read_text())
    fixtures = load_fixtures()
    if len(fixtures) != protocol['fixture_count']:
        raise ValueError('fixture count differs from frozen protocol')
    from hashlib import sha256
    with zipfile.ZipFile(Path(cache) / build['dependencies'][0]['filename']) as jar:
        learner_bytes_sha256 = sha256(jar.read('de/learnlib/algorithm/ttt/mealy/TTTLearnerMealy.class')).hexdigest()
    manifest = {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_revision': 'content-addressed snapshot; API checkout has no local Git metadata',
                'source_sha256': source_hashes(), 'dependencies': build['dependencies'],
                'java_version': build['java_version'], 'compiled_class_sha256': build['class_sha256'],
                'compiled_classes_sha256': build['compiled_classes_sha256'],
                'learnlib_ttt_class_sha256': learner_bytes_sha256,
                'python_version': sys.version, 'platform': platform.platform(),
                'command': 'PYTHONPATH=src python -m experiments.ttt.runner run',
                'fixture_frozen_at_utc': fixture_data['frozen_at_utc']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    trials = []
    for fixture in fixtures:
        reference = build_reference(fixture)
        for repetition in protocol['repetitions']:
            trial = run_trial(fixture, repetition, build)
            trial['hypothesis_sha256'] = object_digest(trial['hypothesis'])
            trial['reference_sha256'] = object_digest(reference)
            trials.append(trial)
            print(f"{fixture['id']} #{repetition}: {trial['status']} states={trial['hypothesis_states']} MQ={trial['mq_count']} EQ={trial['eq_count']}", flush=True)
        (output / 'trials.json').write_text(json.dumps(trials, separators=(',', ':')) + '\n')
    negatives = negative_controls(build_reference(fixtures[0]), trials[0]['hypothesis'])
    report = {'protocol': 'ttt-v0.1', 'manifest': manifest, 'trials': trials,
              'negative_checks': negatives, 'summary': derived_summary(trials, len(fixtures))}
    validation = validate_report(report)
    if any(ce is None for ce in negatives.values()):
        raise ValueError('negative equivalence check accepted a false hypothesis')
    report['independent_validation'] = validation
    (output / 'report.json').write_text(json.dumps(report, separators=(',', ':')) + '\n')
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'validate'])
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/results/ttt-v0.1')
    parser.add_argument('--cache', type=Path, default=Path('/tmp/artifacts'))
    args = parser.parse_args()
    if args.command == 'run':
        print(json.dumps(run_all(args.output, args.cache)['summary'], indent=2))
    else:
        print(json.dumps(validate_report(json.loads((args.output / 'report.json').read_text())), indent=2))


if __name__ == '__main__':
    main()
