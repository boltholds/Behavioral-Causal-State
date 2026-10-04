import json
from pathlib import Path
import numpy as np
import pytest
from bcs.embedding_cache import build_cache, load_cache
from bcs.sonar_encoder import verify_file


class CountingEncoder:
    dimension=3
    identity={'family':'unit-test-fixture'}
    def __init__(self): self.seen=[]
    def encode(self,texts):
        self.seen.extend(texts)
        return np.array([[len(t),sum(t.encode()),1] for t in texts],dtype=np.float32)


def public(path):
    messages=lambda texts:[{'role':'user','speaker_id':'speaker-0','text':t} for t in texts]
    rows=[{'record_id':'a:0','group_id':'a','messages':messages(['x','yy','x','yy'])},
          {'record_id':'b:0','group_id':'b','messages':messages(['yy','x','yy','x'])}]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_dedup_keeps_order_and_reads_public_only(tmp_path):
    source=public(tmp_path/'train.jsonl');encoder=CountingEncoder()
    build_cache(source,encoder,tmp_path/'cache',batch_size=1)
    assert encoder.seen==['x','yy']
    cache=load_cache(tmp_path/'cache',source)
    z,lengths=cache.batch([0,1])
    assert z.shape==(2,16,3) and z.dtype==np.float32
    assert lengths.tolist()==[4,4]
    assert z[0,:4,0].tolist()==[1,2,1,2]
    assert np.all(z[:,4:]==0)
    assert cache.record_ids==('a:0','b:0')


@pytest.mark.parametrize('fault',['text','vectors','records','metadata'])
def test_cache_corruption_or_public_metadata_fails_closed(tmp_path,fault):
    source=public(tmp_path/'train.jsonl');out=tmp_path/'cache'
    if fault=='metadata':
        rows=[json.loads(s) for s in source.read_text().splitlines()];rows[0]['labels']=[1]
        source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with pytest.raises(ValueError):build_cache(source,CountingEncoder(),out)
        return
    build_cache(source,CountingEncoder(),out)
    path=source if fault=='text' else out/('vectors.npy' if fault=='vectors' else 'records.json')
    with path.open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError):load_cache(out,source)


def test_asset_hash_must_match_local_bytes(tmp_path):
    from hashlib import sha256
    p=tmp_path/'asset';p.write_bytes(b'valid')
    verify_file(p,sha256(b'valid').hexdigest(),5)
    with pytest.raises(ValueError):verify_file(p,'0'*64,5)


def test_sonar_label_alone_cannot_verify_encoder():
    from bcs.sonar_encoder import validate_sonar_identity
    with pytest.raises(ValueError):validate_sonar_identity({'family':'SONAR'},2)
