import pytest

torch = pytest.importorskip('torch')
from experiments.l1.runtime import configure_runtime


def test_cpu_entry_restores_deterministic_float32_policy():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    identity = configure_runtime('cpu')
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32
    assert identity['device'] == 'cpu'
    assert identity['precision'] == 'float32_no_amp_no_tf32'
    assert configure_runtime('cpu') == identity


def test_cuda_never_silently_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA unavailable'):
        configure_runtime('cuda')


def test_cuda_requires_cublas_configuration_before_use(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ValueError, match='CUBLAS_WORKSPACE_CONFIG'):
        configure_runtime('cuda')


def test_unsupported_accelerator_rejected():
    with pytest.raises(ValueError, match='CPU or CUDA'):
        configure_runtime('mps')


@pytest.mark.parametrize('kind', ['concat', 'attention', 'gru'])
def test_synthetic_probe_exercises_training_and_replay_on_cpu(kind):
    from experiments.l1.gpu_preflight import probe
    torch.set_num_threads(1)
    configure_runtime('cpu')
    row = probe(kind, 'cpu', batch_size=2, input_dim=4, events=2)
    assert row['status'] == 'passed' and row['optimizer_repeats'] == 2
    assert row['target_tokens'] == 13
    assert row['peak_cuda_allocated_bytes'] is None
