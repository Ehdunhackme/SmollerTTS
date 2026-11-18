"""Final acoustic network. Pretrained HiFi-GAN V2 provides waveform synthesis."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import TextBlock, FrameBlock, duration_indices

@dataclass(frozen=True)
class Config:
    vocabulary:int=132
    width:int=160
    text_layers:int=4
    frame_dilations:tuple=(1,2,4,8,1,2)
    parameter_limit:int=3_500_000


class Acoustic(nn.Module):
    def __init__(self,config=Config()):
        super().__init__();w=config.width
        if w%4:raise ValueError('Width must divide four attention heads')
        self.width=w
        self.phone=nn.Embedding(config.vocabulary,w,padding_idx=0)
        self.stress=nn.Embedding(3,w)
        self.text=nn.ModuleList(TextBlock(w) for _ in range(config.text_layers))
        self.duration_conv=nn.ModuleList([nn.Conv1d(w,w,3,padding=1),nn.Conv1d(w,w,3,padding=1)])
        self.duration_norm=nn.ModuleList([nn.LayerNorm(w),nn.LayerNorm(w)])
        self.duration_out=nn.Linear(w,1)
        self.phone_prosody=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.GELU(),nn.Linear(w,3))
        self.position=nn.Linear(3,w)
        self.frames=nn.ModuleList(FrameBlock(w,expansion=2,dilation=d) for d in config.frame_dilations)
        self.prosody_residual=nn.Linear(w,3)
        self.prosody_condition=nn.Linear(3,w)
        self.mel_out=nn.Linear(w,80)
        nn.init.constant_(self.mel_out.bias,-5.)

    def encode(self,tokens,stress):
        mask=(tokens!=0).unsqueeze(-1).to(self.phone.weight.dtype)
        t=torch.arange(tokens.shape[1],device=tokens.device,dtype=torch.float32)
        freq=torch.exp(torch.arange(0,self.width,2,device=tokens.device)*(-math.log(10000)/self.width))
        phase=t[:,None]*freq[None]
        position=torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        x=(self.phone(tokens)+self.stress(stress)+position[None])*mask
        for block in self.text:x=block(x,mask)
        duration=x
        for conv,norm in zip(self.duration_conv,self.duration_norm):
            duration=F.relu(norm(conv(duration.transpose(1,2)).transpose(1,2)))*mask
        return x,self.duration_out(duration).squeeze(-1)*mask.squeeze(-1),self.phone_prosody(x)*mask

    def decode(self,encoded,phone_prosody,indices,position,frame_mask):
        # Length regulation is outside the neural graph; browser JS can supply indices.
        x=torch.gather(encoded,1,indices[...,None].expand(-1,-1,self.width))
        coarse=torch.gather(phone_prosody,1,indices[...,None].expand(-1,-1,3))
        x=(x+self.position(torch.stack((position,torch.sin(math.pi*position),torch.cos(math.pi*position)),-1)))*frame_mask
        for block in self.frames[:2]:x=block(x,frame_mask)
        prosody=(coarse+self.prosody_residual(x))*frame_mask
        condition=torch.stack((prosody[...,0],prosody[...,1].sigmoid(),prosody[...,2]),-1)
        x=(x+self.prosody_condition(condition))*frame_mask
        for block in self.frames[2:]:x=block(x,frame_mask)
        return self.mel_out(x)*frame_mask,prosody


class AcousticModel(Acoustic):
    def __init__(self, mean=None, std=None):
        super().__init__(Config())
        nn.init.zeros_(self.mel_out.bias)
        self.register_buffer('mel_mean', torch.zeros(80) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('mel_std', torch.ones(80) if std is None else torch.as_tensor(std, dtype=torch.float32))

    def native_decode(self, encoded, phone_prosody, indices, position, frame_mask):
        normalized, prosody = self.decode(encoded, phone_prosody, indices, position, frame_mask)
        return normalized*self.mel_std+self.mel_mean, prosody


    def decode_recorded(self, encoded, phone_prosody, indices, position, frame_mask, recorded):
        x = torch.gather(encoded, 1, indices[..., None].expand(-1, -1, self.width))
        coarse = torch.gather(phone_prosody, 1, indices[..., None].expand(-1, -1, 3))
        x = (x + self.position(torch.stack((position, torch.sin(math.pi*position),
                                            torch.cos(math.pi*position)), -1))) * frame_mask
        for block in self.frames[:2]:
            x = block(x, frame_mask)
        prosody = (coarse + self.prosody_residual(x)) * frame_mask
        if recorded.shape != prosody.shape:
            raise ValueError('Recorded conditioning shape mismatch')
        # Native normalized log-F0, binary voicing, normalized log energy.
        # Do not sigmoid the recorded 0/1 voicing channel or train through labels.
        x = (x + self.prosody_condition(recorded.detach())) * frame_mask
        for block in self.frames[2:]:
            x = block(x, frame_mask)
        return self.mel_out(x) * frame_mask, prosody
