"""Frozen public-text SONAR preparation; no labels, fitting, or normalization."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from bcs.embedding_cache import build_cache, load_cache, read_public
from bcs.sonar_encoder import SonarEncoder, file_hash, validate_sonar_identity

SPLITS=('train','validation','iid_test','challenge')


class MemoizedEncoder:
    def __init__(self,encoder):
        self.encoder=encoder;self.dimension=encoder.dimension;self.identity=encoder.identity
        self.vectors={};self.encoded=0

    def seed(self,texts,matrix):
        for text,vector in zip(texts,matrix):
            if text in self.vectors and not np.array_equal(self.vectors[text],vector):
                raise ValueError('inconsistent verified vectors across splits')
            self.vectors[text]=vector

    def encode(self,texts):
        missing=tuple(dict.fromkeys(t for t in texts if t not in self.vectors))
        if missing:
            vectors=self.encoder.encode(missing)
            if vectors.dtype!=np.float32 or vectors.shape!=(len(missing),self.dimension) or not np.isfinite(vectors).all():
                raise ValueError('encoder violated float32 sentence contract')
            self.seed(missing,vectors);self.encoded+=len(missing)
        return np.stack([self.vectors[t] for t in texts])


def prepare(dataset,encoder,output,batch_size=16,progress=lambda event:None):
    if batch_size<1:raise ValueError('positive batch size required')
    started=time.monotonic();dataset=Path(dataset);output=Path(output)
    public={split:dataset/'public'/f'{split}.jsonl' for split in SPLITS}
    # Validate every public input and existing destination before starting new work.
    texts={split:tuple(dict.fromkeys(m['text'] for r in read_public(path) for m in r['messages'])) for split,path in public.items()}
    memo=MemoizedEncoder(encoder);existing={}
    for split,path in public.items():
        destination=output/split
        if (output/f'.{split}.partial').exists():raise ValueError(f'unverified partial output: {split}')
        if destination.exists():
            cache=load_cache(destination,path)
            if cache.manifest['encoder']!=encoder.identity or cache.manifest['dimension']!=encoder.dimension:
                raise ValueError(f'cache encoder identity mismatch: {split}')
            memo.seed(texts[split],cache.vectors);existing[split]=cache.manifest
    output.mkdir(parents=True,exist_ok=True);summary={}
    for split,path in public.items():
        split_start=time.monotonic();before=memo.encoded
        if split in existing:manifest=existing[split]
        else:
            staging=output/f'.{split}.partial'
            manifest=build_cache(path,memo,staging,batch_size=batch_size,
                progress=lambda e:progress({'split':split,'global_encoded':memo.encoded,**e}))
            load_cache(staging,path)
            staging.rename(output/split)
        summary[split]={**manifest,'manifest_sha256':file_hash(output/split/'manifest.json'),
                        'resumed':split in existing,'newly_encoded':memo.encoded-before,
                        'elapsed_seconds':time.monotonic()-split_start}
        progress({'split':split,'complete':True,'records':manifest['records'],'global_encoded':memo.encoded})
    return {'schema':'l1-preparation-v1','encoder':encoder.identity,'splits':summary,
            'global_unique_sentences':len(memo.vectors),'newly_encoded':memo.encoded,
            'elapsed_seconds':time.monotonic()-started,'preprocessing':'none; exact text memoization only',
            'label_access':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--assets',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.threads<1 or args.batch_size<1:parser.error('threads and batch size must be positive')
    import torch
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(args.threads)
    encoder=SonarEncoder(args.assets,batch_size=args.batch_size)
    validate_sonar_identity(encoder.identity,encoder.dimension)
    last=[0.0]
    def progress(event):
        now=time.monotonic()
        if event.get('complete') or now-last[0]>=20:
            print(json.dumps({'elapsed_seconds':now-start,**event}),flush=True);last[0]=now
    start=time.monotonic()
    result=prepare(args.dataset,encoder,args.output,batch_size=args.batch_size,progress=progress)
    result['threads']=args.threads;result['batch_size']=args.batch_size
    result['dataset_manifest_sha256']=file_hash(Path(args.dataset)/'manifest.json')
    result['catalog_sha256']=file_hash(Path(args.dataset)/'catalog.json')
    (Path(args.output)/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'newly_encoded':result['newly_encoded'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
