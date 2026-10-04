"""Bind a diagnostic prefix subset to audited parent data before training."""
from pathlib import Path
from hashlib import sha256
import json
from itertools import islice
from .sonar_encoder import file_hash,validate_sonar_identity
from .embedding_cache import read_public,load_cache

DATA_FILES=('public/train.jsonl','public/validation.jsonl','training/labels.jsonl','evaluator/validation.jsonl','core.json')
CACHE_FILES=tuple(f'{split}/{name}' for split in ('train','validation') for name in ('manifest.json','records.json','vectors.npy'))


def _rows(path):
    with Path(path).open() as file:return [json.loads(line) for line in file]


def _prefix(path,count):
    with Path(path).open() as file:return list(islice(file,count))


def source_hashes():
    return {p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare_subset(parent,output,train_count=128,validation_count=32):
    parent,output=Path(parent),Path(output)
    if any(type(x) is not int or x<1 for x in (train_count,validation_count)):
        raise ValueError('positive subset counts required')
    if output.exists() and any(output.iterdir()):raise FileExistsError('subset destination must be empty')
    for folder in ('public','training','evaluator'):(output/folder).mkdir(parents=True,exist_ok=True)
    for split,count in (('train',train_count),('validation',validation_count)):
        rows=_prefix(parent/f'public/{split}.jsonl',count)
        if len(rows)!=count:raise ValueError('subset exceeds parent support')
        (output/f'public/{split}.jsonl').write_text(''.join(rows))
    ids={r['record_id'] for r in read_public(output/'public/train.jsonl')}
    labels=[r for r in _rows(parent/'training/labels.jsonl') if r['record_id'] in ids]
    (output/'training/labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in labels))
    (output/'evaluator/validation.jsonl').write_text(''.join(_prefix(parent/'evaluator/validation.jsonl',validation_count)))
    (output/'core.json').write_bytes((parent/'core.json').read_bytes())


def create_input_lock(dataset,parent,cache_root,protocol,progress=lambda x:None):
    from .dataset import audit_dataset
    dataset,parent,cache_root=Path(dataset),Path(parent),Path(cache_root)
    if (dataset/'input-lock.json').exists():raise FileExistsError('input lock already exists; never silently relock')
    audit=audit_dataset(parent,progress)
    if audit['integrity_status']!='passed':raise ValueError('parent dataset failed audit: '+str(audit['errors']))
    counts={}
    for split in ('train','validation'):
        rows=read_public(dataset/f'public/{split}.jsonl');counts[split]=len(rows)
        if rows!=[json.loads(line) for line in _prefix(parent/f'public/{split}.jsonl',len(rows))]:
            raise ValueError('public subset differs from fixed parent prefix')
        cache=load_cache(cache_root/split,dataset/f'public/{split}.jsonl')
        validate_sonar_identity(cache.manifest['encoder'],cache.vectors.shape[1])
    ids={r['record_id'] for r in read_public(dataset/'public/train.jsonl')}
    expected=[r for r in _rows(parent/'training/labels.jsonl') if r['record_id'] in ids]
    if _rows(dataset/'training/labels.jsonl')!=expected:
        raise ValueError('training labels differ from audited parent')
    if _rows(dataset/'evaluator/validation.jsonl')!=[json.loads(l) for l in _prefix(parent/'evaluator/validation.jsonl',counts['validation'])]:
        raise ValueError('validation labels differ from audited parent')
    if (dataset/'core.json').read_bytes()!=(parent/'core.json').read_bytes():
        raise ValueError('frozen core differs from audited parent')
    lock={'schema':'language-input-lock-v1','scope':'diagnostic','selection':'fixed_parent_prefix','counts':counts,
          'parent_manifest_sha256':file_hash(parent/'manifest.json'),'parent_integrity':'passed',
          'dataset_files':{p:file_hash(dataset/p) for p in DATA_FILES},
          'cache_files':{p:file_hash(cache_root/p) for p in CACHE_FILES},
          'protocol_sha256':file_hash(protocol),'source_sha256':source_hashes()}
    (dataset/'input-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    return file_hash(dataset/'input-lock.json')


def verify_input_lock(dataset,cache_root,protocol):
    dataset,cache_root=Path(dataset),Path(cache_root)
    path=dataset/'input-lock.json'
    if not path.is_file():raise ValueError('missing input-lock.json; create a verified diagnostic lock first')
    lock=json.loads(path.read_text())
    if lock['schema']!='language-input-lock-v1' or lock['scope']!='diagnostic' or lock['parent_integrity']!='passed':
        raise ValueError('unsupported input lock')
    if set(lock['dataset_files'])!=set(DATA_FILES) or set(lock['cache_files'])!=set(CACHE_FILES):
        raise ValueError('input lock file set mismatch')
    if lock['protocol_sha256']!=file_hash(protocol) or lock['source_sha256']!=source_hashes():
        raise ValueError('protocol/source differs from input lock')
    for key,root in (('dataset_files',dataset),('cache_files',cache_root)):
        for name,digest in lock[key].items():
            if file_hash(root/name)!=digest:raise ValueError(f'locked input changed: {name}')
    for split in ('train','validation'):
        cache=load_cache(cache_root/split,dataset/f'public/{split}.jsonl')
        validate_sonar_identity(cache.manifest['encoder'],cache.vectors.shape[1])
        if len(cache.record_ids)!=lock['counts'][split]:raise ValueError('locked count mismatch')
    return file_hash(path)
