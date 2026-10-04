"""Public-text-only sentence cache. No evaluator files are accepted by this API."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .sonar_encoder import SentenceEncoder, file_hash


def read_public(path):
    records = []
    ids = set()
    with Path(path).open() as file:
        for line in file:
            row = json.loads(line)
            if set(row) != {'record_id', 'group_id', 'messages'} or not all(isinstance(row[k], str) for k in ('record_id', 'group_id')):
                raise ValueError('public record boundary violated')
            if row['record_id'] in ids or not 1 <= len(row['messages']) <= 16:
                raise ValueError('duplicate record or unsupported length')
            ids.add(row['record_id'])
            for m in row['messages']:
                if set(m) != {'role', 'speaker_id', 'text'} or m['role'] != 'user' or m['speaker_id'] != 'speaker-0' or not isinstance(m['text'], str) or not m['text']:
                    raise ValueError('unsupported message metadata/text')
            records.append(row)
    if not records:
        raise ValueError('empty public dataset')
    return records


def build_cache(public_path, encoder: SentenceEncoder, output, batch_size=32, progress=lambda x: None):
    if batch_size < 1:
        raise ValueError('positive batch size required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('cache destination must be empty')
    rows = read_public(public_path)
    texts = {}; records = []
    for row in rows:
        indices = []
        for message in row['messages']:
            text = message['text']
            if text not in texts:
                texts[text] = len(texts)
            indices.append(texts[text])
        records.append({'record_id': row['record_id'], 'group_id': row['group_id'], 'indices': indices})
    output.mkdir(parents=True, exist_ok=True)
    sentences = tuple(texts)
    matrix = np.lib.format.open_memmap(output/'vectors.npy', mode='w+', dtype=np.float32, shape=(len(texts), encoder.dimension))
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start+batch_size]
        vectors = encoder.encode(chunk)
        if vectors.dtype != np.float32 or vectors.shape != (len(chunk), encoder.dimension) or not np.isfinite(vectors).all():
            raise ValueError('encoder violated float32 sentence contract')
        matrix[start:start+len(chunk)] = vectors
        progress({'encoded_unique': start+len(chunk), 'unique_total': len(sentences)})
    matrix.flush()
    (output/'records.json').write_text(json.dumps(records, separators=(',', ':'))+'\n')
    manifest = {'schema': 'embedding-cache-v1', 'public_sha256': file_hash(public_path),
                'encoder': encoder.identity, 'dimension': encoder.dimension, 'dtype': 'float32',
                'unique_sentences': len(texts), 'records': len(records),
                'files': {n: file_hash(output/n) for n in ('vectors.npy', 'records.json')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


@dataclass(frozen=True)
class EmbeddingCache:
    vectors: np.ndarray
    record_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    indices: tuple[tuple[int, ...], ...]
    manifest: dict  # external serialized provenance, never passed to the model

    def batch(self, rows):
        rows = tuple(rows)
        z = np.zeros((len(rows), 16, self.vectors.shape[1]), dtype=np.float32)
        lengths = np.array([len(self.indices[i]) for i in rows], dtype=np.int64)
        for j, i in enumerate(rows):
            z[j, :lengths[j]] = self.vectors[list(self.indices[i])]
        return z, lengths


def load_cache(output, public_path):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['schema'] != 'embedding-cache-v1' or manifest['public_sha256'] != file_hash(public_path) or manifest['dtype'] != 'float32':
        raise ValueError('cache source/schema mismatch')
    if set(manifest['files']) != {'vectors.npy', 'records.json'}:
        raise ValueError('cache file set mismatch')
    for name, digest in manifest['files'].items():
        if file_hash(output/name) != digest:
            raise ValueError('cache checksum mismatch')
    rows = json.loads((output/'records.json').read_text())
    public = read_public(public_path)
    matrix = np.load(output/'vectors.npy', mmap_mode='r', allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (manifest['unique_sentences'], manifest['dimension']) or not np.isfinite(matrix).all():
        raise ValueError('cache matrix contract violated')
    expected = {}; indices = []
    for row in public:
        current = []
        for m in row['messages']:
            if m['text'] not in expected: expected[m['text']] = len(expected)
            current.append(expected[m['text']])
        indices.append({'record_id': row['record_id'], 'group_id': row['group_id'], 'indices': current})
    if rows != indices or len(rows) != manifest['records'] or len(expected) != len(matrix):
        raise ValueError('cache index/public alignment mismatch')
    return EmbeddingCache(matrix, tuple(r['record_id'] for r in rows), tuple(r['group_id'] for r in rows),
                          tuple(tuple(r['indices']) for r in rows), manifest)
