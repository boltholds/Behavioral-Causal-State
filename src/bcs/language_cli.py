"""SONAR cache and diagnostic neural training. No command declares L1 passed."""
import argparse
import json
from pathlib import Path
import sys


def validation_events(path,cache):
    from .dataset_codec import event_from_wire
    events={}
    with Path(path).open() as file:
        for line in file:
            row=json.loads(line)
            if row['kind']!='base' or len(row['sides'])!=1:
                raise ValueError('base validation sidecar required')
            side=row['sides'][0];rid=side['record_id']
            if rid in events:raise ValueError('duplicate validation record')
            events[rid]=tuple(event_from_wire(e) for e in side['events'])
    if set(events)!=set(cache.record_ids):
        raise ValueError('validation sidecar/cache ID mismatch')
    return tuple(events[rid] for rid in cache.record_ids)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    subset=commands.add_parser('subset',help='copy a fixed diagnostic prefix; lock must be created after encoding')
    subset.add_argument('--parent',type=Path,required=True)
    subset.add_argument('--output',type=Path,required=True)
    subset.add_argument('--train-count',type=int,default=128)
    subset.add_argument('--validation-count',type=int,default=32)
    lock=commands.add_parser('lock',help='audit parent and verify subset/cache before locking')
    lock.add_argument('--dataset',type=Path,required=True)
    lock.add_argument('--parent',type=Path,required=True)
    lock.add_argument('--cache-root',type=Path,required=True)
    cache=commands.add_parser('cache',help='encode public messages using verified local SONAR assets')
    cache.add_argument('--public',type=Path,required=True)
    cache.add_argument('--assets',type=Path,required=True)
    cache.add_argument('--output',type=Path,required=True)
    cache.add_argument('--batch-size',type=int,default=16)
    train=commands.add_parser('train',description='Single diagnostic configuration; registered L1 acceptance is not run.')
    train.add_argument('--dataset',type=Path,required=True)
    train.add_argument('--cache-root',type=Path,required=True)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--branch',choices=('concat','attention','gru'),required=True)
    train.add_argument('--epochs',type=int,default=30)
    train.add_argument('--batch-size',type=int,default=64)
    train.add_argument('--learning-rate',type=float,default=.0003)
    train.add_argument('--seed',type=int,default=11)
    for command in (cache,train):
        command.add_argument('--device',default='cpu')
        command.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    try:
        protocol=Path(__file__).resolve().parents[2]/'experiments/protocols/language-v0.1.json'
        from .language_lock import prepare_subset,create_input_lock,verify_input_lock
        if args.command=='subset':
            prepare_subset(args.parent,args.output,args.train_count,args.validation_count)
            print(json.dumps({'output':str(args.output),'scope':'diagnostic_unlocked'}));return 0
        if args.command=='lock':
            digest=create_input_lock(args.dataset,args.parent,args.cache_root,protocol,lambda r:print(json.dumps(r),file=sys.stderr,flush=True))
            print(json.dumps({'input_lock_sha256':digest,'scope':'diagnostic'}));return 0
        import torch
        if args.threads<1:raise ValueError('positive thread count required')
        torch.set_num_threads(args.threads)
        from .embedding_cache import build_cache,load_cache
        def progress(value):print(json.dumps(value),file=sys.stderr,flush=True)
        if args.command=='cache':
            from .sonar_encoder import SonarEncoder
            encoder=SonarEncoder(args.assets,args.device,args.batch_size)
            report=build_cache(args.public,encoder,args.output,args.batch_size,progress)
            print(json.dumps(report));return 0
        from .language_training import TrainConfig,train_one,load_labels,load_checkpoint,predict_cache
        from .grounder import ReaderKind
        from .language_tokens import encode_events
        from .language_evaluation import causal_report
        from .calibration import BooleanCore
        from .sonar_encoder import file_hash
        input_lock=verify_input_lock(args.dataset,args.cache_root,protocol)
        train_cache=load_cache(args.cache_root/'train',args.dataset/'public/train.jsonl')
        valid_cache=load_cache(args.cache_root/'validation',args.dataset/'public/validation.jsonl')
        labels=load_labels(args.dataset/'training/labels.jsonl',train_cache)
        events=validation_events(args.dataset/'evaluator/validation.jsonl',valid_cache)
        gold=tuple(encode_events(e) for e in events)
        config=TrainConfig(args.epochs,args.batch_size,args.learning_rate,args.seed)
        report=train_one(ReaderKind(args.branch),train_cache,labels,valid_cache,gold,args.output,config,args.device,progress,input_lock_sha256=input_lock)
        if verify_input_lock(args.dataset,args.cache_root,protocol)!=input_lock:raise ValueError('input lock changed during training')
        model=load_checkpoint(args.output,train_cache.manifest['encoder'],input_lock,args.device)
        predictions=predict_cache(model,valid_cache,args.batch_size,args.device)
        core=BooleanCore.from_wire(json.loads((args.dataset/'core.json').read_text()))
        report['causal_validation']=causal_report(predictions,events,core)
        report['input_hashes']={name:file_hash(args.dataset/name) for name in ('public/train.jsonl','public/validation.jsonl','training/labels.jsonl','evaluator/validation.jsonl','core.json')}
        report['protocol_sha256']=file_hash(protocol)
        report['input_lock_sha256']=input_lock
        report['encoder_identity_matches_pinned_contract']=True
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        with (args.output/'validation-predictions.jsonl').open('w') as file:
            for rid,tokens in zip(valid_cache.record_ids,predictions):
                file.write(json.dumps({'record_id':rid,'tokens':tokens})+'\n')
        print(json.dumps({'output':str(args.output),'branch':args.branch,'L1_status':report['L1_status'],'validation_history_error':report['validation']['full_history_error'],'language_tv':report['causal_validation']['language_mean_tv']}))
        return 0
    except (ValueError,OSError,ImportError,KeyError) as error:
        parser.exit(2,f'error: {error}\n')


if __name__=='__main__':
    raise SystemExit(main())
