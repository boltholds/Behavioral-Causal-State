"""Synthetic CUDA execution/replay check; reads no experiment data or labels."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import gc
import json
import torch
from bcs.generator import Executed
from bcs.grounder import Grounder, ReaderKind
from bcs.language_tokens import encode_events
from bcs.simulator import Action
from .runtime import configure_runtime
from .training import atomic_json, _versions


def probe(kind, device, *, batch_size=64, input_dim=1024, events=16):
    signatures = []
    started = perf_counter()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        torch.manual_seed(8128)
        z = torch.randn(batch_size, 16, input_dim).to(device)
        lengths = torch.full((batch_size,), 16, dtype=torch.long, device=device)
        tokens = encode_events(Executed(t + 1, 0, Action.START) for t in range(events))
        target = torch.tensor([tokens] * batch_size, device=device)
        model = Grounder(ReaderKind(kind), input_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        logits = model.teacher_logits(z, lengths, target)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite CUDA probe loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        model.eval()
        prediction = model.greedy(z[:2], lengths[:2])
        if prediction != model.greedy(z[:2], lengths[:2]):
            raise ValueError(f'{kind}: greedy replay differs')
        weights = sha256()
        for name, tensor in sorted(model.state_dict().items()):
            weights.update(name.encode())
            weights.update(tensor.detach().cpu().numpy().tobytes())
        signatures.append((float(loss.detach()), weights.hexdigest(), prediction))
        del model, optimizer, logits, loss, z, lengths, target
        gc.collect()
    if signatures[0] != signatures[1]:
        raise ValueError(f'{kind}: seeded optimizer replay differs')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    return {'branch': kind, 'status': 'passed', 'batch_size': batch_size,
            'input_dim': input_dim, 'history_length': 16, 'target_tokens': len(tokens),
            'optimizer_repeats': 2, 'loss': signatures[0][0],
            'seconds': perf_counter() - started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = configure_runtime('cuda')
    rows = []
    for kind in ('concat', 'attention', 'gru'):
        row = probe(kind, 'cuda')
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {'scope': 'synthetic_execution_not_L1', 'status': 'passed',
        'runtime': runtime, 'versions': _versions(), 'branches': rows,
        'limits': 'Tests seeded training and greedy replay, not cross-machine equality, epoch resume or language quality.'})


if __name__ == '__main__':
    main()
