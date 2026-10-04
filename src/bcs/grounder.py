"""SPEC-03 readers and shared autoregressive structured-event decoder."""
from enum import StrEnum
import torch
from torch import nn
from .language_tokens import PAD, BOS, EOS, VOCAB_SIZE, MAX_TOKENS, _choices


class ReaderKind(StrEnum):
    CONCAT = 'concat'
    ATTENTION = 'attention'
    GRU = 'gru'


class Grounder(nn.Module):
    def __init__(self, kind: ReaderKind, input_dim=1024, dropout=0.1):
        super().__init__()
        if not isinstance(kind, ReaderKind) or input_dim < 1:
            raise ValueError('typed reader and positive input dimension required')
        self.kind, self.input_dim = kind, input_dim
        if kind == ReaderKind.CONCAT:
            self.reader = nn.Sequential(nn.Linear(16*input_dim+17,256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256,256), nn.GELU())
        elif kind == ReaderKind.GRU:
            self.reader = nn.GRU(input_dim,256,batch_first=True)
        else:
            self.projection = nn.Linear(input_dim,256)
            self.positions = nn.Embedding(16,256)
            layer = nn.TransformerEncoderLayer(256,8,1024,dropout,batch_first=True,activation='gelu')
            self.reader = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.cross_attention = nn.MultiheadAttention(256,8,dropout=dropout,batch_first=True)
        self.tokens = nn.Embedding(VOCAB_SIZE,64,padding_idx=PAD)
        self.decoder = nn.GRU(64+256,256,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(256,VOCAB_SIZE)

    def _read(self,z,lengths):
        if z.ndim != 3 or z.shape[1:] != (16,self.input_dim) or lengths.shape != (len(z),) or lengths.dtype != torch.int64 or not len(z):
            raise ValueError('expected B x 16 x input_dim and int64 lengths')
        if torch.any((lengths<1)|(lengths>16)):
            raise ValueError('history length must be 1..16')
        padding = torch.arange(16,device=z.device)[None,:] >= lengths[:,None]
        z = z.masked_fill(padding[:,:,None],0)
        if not torch.isfinite(z).all():
            raise ValueError('nonfinite sentence embedding')
        if self.kind == ReaderKind.CONCAT:
            context = self.reader(torch.cat((z.flatten(1),(~padding).to(z.dtype),lengths[:,None].to(z.dtype)/16),dim=-1))
            memory = context[:,None,:]
        elif self.kind == ReaderKind.GRU:
            packed = nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _, hidden = self.reader(packed)
            context = hidden[0]; memory = context[:,None,:]
        else:
            memory = self.reader(self.projection(z)+self.positions(torch.arange(16,device=z.device))[None,:,:],src_key_padding_mask=padding)
            context = memory.masked_fill(padding[:,:,None],0).sum(1)/lengths[:,None]
        return context,memory,padding

    def _logits(self,decoded,memory,padding):
        if self.kind == ReaderKind.ATTENTION:
            attended,_ = self.cross_attention(decoded,memory,memory,key_padding_mask=padding,need_weights=False)
            decoded = decoded+attended
        return self.output(self.dropout(decoded))

    def teacher_logits(self,z,lengths,targets):
        """Teacher forcing for loss only; input at step i contains targets[:i]."""
        context,memory,padding = self._read(z,lengths)
        if targets.ndim != 2 or targets.shape[0] != len(z) or not 1 <= targets.shape[1] <= MAX_TOKENS:
            raise ValueError('invalid target shape')
        shifted = torch.cat((torch.full_like(targets[:,:1],BOS),targets[:,:-1]),dim=1)
        x = torch.cat((self.tokens(shifted),context[:,None,:].expand(-1,targets.shape[1],-1)),dim=-1)
        decoded,_ = self.decoder(x,context[None,:,:])
        logits = self._logits(decoded,memory,padding)
        valid = torch.zeros(logits.shape,dtype=torch.bool,device=z.device)
        for row,tokens in enumerate(targets.detach().cpu().tolist()):
            prefix=[]; ended=False
            for step,token in enumerate(tokens):
                choices=(PAD,) if ended else _choices(prefix)
                if token not in choices:
                    raise ValueError('teacher target violates structured grammar')
                valid[row,step,list(choices)]=True
                ended=ended or token==EOS
                prefix.append(token)
        return logits.masked_fill(~valid,torch.finfo(logits.dtype).min)

    @torch.inference_mode()
    def greedy(self,z,lengths):
        if self.training:
            raise ValueError('greedy evaluation requires model.eval()')
        context,memory,padding = self._read(z,lengths)
        hidden=context[None,:,:]
        previous=torch.full((len(z),1),BOS,dtype=torch.long,device=z.device)
        sequences=[[] for _ in range(len(z))]; ended=[False]*len(z)
        for _ in range(MAX_TOKENS):
            x=torch.cat((self.tokens(previous),context[:,None,:]),dim=-1)
            decoded,hidden=self.decoder(x,hidden)
            logits=self._logits(decoded,memory,padding)[:,0,:]
            valid=torch.zeros_like(logits,dtype=torch.bool)
            for i,prefix in enumerate(sequences):
                valid[i,list((PAD,) if ended[i] else _choices(prefix))]=True
            previous=logits.masked_fill(~valid,torch.finfo(logits.dtype).min).argmax(-1,keepdim=True)
            for i,token in enumerate(previous[:,0].cpu().tolist()):
                if not ended[i]:
                    sequences[i].append(token);ended[i]=token==EOS
            if all(ended):break
        return tuple(tuple(tokens) for tokens in sequences)
