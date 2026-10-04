from pathlib import Path
import json,time,torch
from bcs.grounder import Grounder,ReaderKind
from bcs.embedding_cache import load_cache
from bcs.language_training import load_labels,_batch
from bcs.language_tokens import PAD
p=Path('artifacts/language-diagnostic-v0.1');cache=load_cache(p/'embeddings/train',p/'public/train.jsonl');targets=load_labels(p/'training/labels.jsonl',cache)
rows=[]
for threads in (1,2,4):
 torch.set_num_threads(threads)
 for kind in ReaderKind:
  torch.manual_seed(11);m=Grounder(kind);opt=torch.optim.AdamW(m.parameters(),lr=.0003)
  z,n,t=_batch(cache,targets,range(64),'cpu')
  times=[]
  for i in range(6):
   m.train();tic=time.perf_counter();opt.zero_grad(set_to_none=True);a=m.teacher_logits(z,n,t);loss=torch.nn.functional.cross_entropy(a.flatten(0,1),t.flatten(),ignore_index=PAD);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();elapsed=time.perf_counter()-tic
   if i>=2:times.append(elapsed)
  m.eval();tic=time.perf_counter();m.greedy(z,n);greedy=time.perf_counter()-tic
  row={'branch':kind.value,'threads':threads,'training_step_seconds':sum(times)/len(times),'greedy_batch64_seconds':greedy,'maximum_training_only_hours_per_run':sum(times)/len(times)*313*30/3600}
  rows.append(row);print(json.dumps(row),flush=True)
Path('/tmp/bcs-l1-resource-probe.json').write_text(json.dumps(rows,indent=2))
