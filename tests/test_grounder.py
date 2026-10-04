import pytest
torch=pytest.importorskip('torch')
from bcs.grounder import Grounder, ReaderKind
from bcs.language_tokens import encode_events, PAD
from bcs.generator import Executed
from bcs.simulator import Action

torch.set_num_threads(2)


@pytest.mark.parametrize('kind',list(ReaderKind))
def test_padding_does_not_change_logits_and_gradients_exist(kind):
    torch.manual_seed(1)
    model=Grounder(kind,8).eval()
    z=torch.randn(2,16,8); lengths=torch.tensor([4,8])
    tokens=torch.tensor([encode_events((Executed(1,0,Action.START),))]*2)
    a=model.teacher_logits(z,lengths,tokens)
    changed=z.clone();changed[0,4:]=1000;changed[1,8:]=-1000
    b=model.teacher_logits(changed,lengths,tokens)
    assert torch.allclose(a,b,atol=1e-6)
    loss=torch.nn.functional.cross_entropy(a.reshape(-1,a.shape[-1]),tokens.flatten())
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())


@pytest.mark.parametrize('kind',list(ReaderKind))
def test_order_and_future_target_causality(kind):
    torch.manual_seed(11)
    model=Grounder(kind,8).eval();z=torch.randn(1,16,8);n=torch.tensor([4])
    a=torch.tensor([encode_events((Executed(1,0,Action.START),))]);b=torch.tensor([encode_events((Executed(1,0,Action.STOP),))])
    first=model.teacher_logits(z,n,a);second=model.teacher_logits(z,n,b)
    assert torch.equal(first[:,:6],second[:,:6])
    swapped=z.clone();swapped[:,[0,1]]=swapped[:,[1,0]]
    assert not torch.allclose(first,model.teacher_logits(swapped,n,a))
    decoded=model.greedy(z,n)
    assert len(decoded)==1 and 1<=len(decoded[0])<=97


def test_invalid_length_rejected_and_teacher_padding_ignored():
    model=Grounder(ReaderKind.GRU,8).eval()
    z=torch.zeros(1,16,8)
    with pytest.raises(ValueError):model.greedy(z,torch.tensor([17]))
    target=torch.tensor([encode_events((Executed(1,0,Action.START),))+(PAD,)*6])
    assert model.teacher_logits(z,torch.tensor([4]),target).shape==(1,13,40)
