"""Verified local assets for the official SONAR text inference pipeline.

Importing this module does not import torch/fairseq2. The encoder may run in
a separate environment and exchange float32 caches with the training runtime.
"""
from hashlib import sha256
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np

UPSTREAM_REVISION = 'a551c586dcf4a49c8fd847de369412d556a7f2f2'
ASSETS = (
    ('sonar_text_encoder.pt', 'b46ffc66abd0519ce6693797eaf8b15da9011159eae18d99900c273a54208bf7', 3064513202),
    ('sentencepiece.source.256000.model', '14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a', 4852054),
)


def pinned_identity():
    return {'family':'SONAR','model':'text_sonar_basic_encoder','dimension':1024,
            'source_lang':'rus_Cyrl','dtype':'float32','normalization':'none',
            'hf_revision':UPSTREAM_REVISION,'checkpoint_sha256':ASSETS[0][1],
            'tokenizer_sha256':ASSETS[1][1]}


def validate_sonar_identity(identity,dimension):
    if dimension!=1024 or any(identity.get(k)!=v for k,v in pinned_identity().items()):
        raise ValueError('encoder identity differs from pinned SONAR contract')


class SentenceEncoder(Protocol):
    dimension: int
    @property
    def identity(self) -> dict: ...  # JSON manifest at the external boundary
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected_hash, expected_size):
    if Path(path).stat().st_size != expected_size or file_hash(path) != expected_hash:
        raise ValueError(f'asset checksum/size mismatch: {Path(path).name}')


class SonarEncoder:
    dimension = 1024

    def __init__(self, asset_dir, device='cpu', batch_size=16):
        import importlib.metadata
        import torch
        import sonar  # register official model families
        from fairseq2.assets import AssetCard, get_asset_store
        from fairseq2.models import load_model
        from fairseq2.data.tokenizers import load_tokenizer
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        root = Path(asset_dir).resolve()
        for name, digest, size in ASSETS:
            verify_file(root/name, digest, size)
        base = get_asset_store().retrieve_card('text_sonar_basic_encoder')
        card = AssetCard('bcs_sonar_locked', {
            'checkpoint': (root/ASSETS[0][0]).as_uri(),
            'tokenizer': (root/ASSETS[1][0]).as_uri(),
        }, base)
        model = load_model(card, device=torch.device(device), dtype=torch.float32, mmap=True)
        tokenizer = load_tokenizer(card)
        self.pipeline = TextToEmbeddingModelPipeline(model, tokenizer, device=torch.device(device), dtype=torch.float32)
        self.pipeline.eval().requires_grad_(False)
        self.batch_size = batch_size
        self.identity = {
            'family': 'SONAR', 'model': 'text_sonar_basic_encoder', 'dimension': 1024,
            'source_lang': 'rus_Cyrl', 'dtype': 'float32', 'normalization': 'none',
            'hf_revision': UPSTREAM_REVISION, 'checkpoint_sha256': ASSETS[0][1],
            'tokenizer_sha256': ASSETS[1][1], 'device': device,
            'versions': {p: importlib.metadata.version(p) for p in ('sonar-space', 'fairseq2', 'fairseq2n', 'torch', 'numpy')},
        }

    def encode(self, texts):
        import torch
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with torch.inference_mode():
            z = self.pipeline.predict(list(texts), source_lang='rus_Cyrl', batch_size=self.batch_size)
        result = z.detach().cpu().float().numpy()
        if result.shape != (len(texts), 1024) or not np.isfinite(result).all():
            raise ValueError('invalid SONAR output')
        return result
