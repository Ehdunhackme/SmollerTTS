"""Recorded-prosody acoustic training objective; inference uses predicted prosody."""
import torch
from torch.nn import functional as F

def masked(value,mask):
    while mask.ndim<value.ndim:mask=mask.unsqueeze(-1)
    return (value*mask).sum()/mask.expand_as(value).sum().clamp_min(1)


def acoustic_objective(model,b,recorded=True):
    e,d,p=model.encode(b['tokens'],b['stress'])
    if recorded:
        m,pr=model.decode_recorded(e,p,b['indices'],b['position'],b['frame_mask'],b['prosody'])
    else:
        m,pr=model.decode(e,p,b['indices'],b['position'],b['frame_mask'])
    target=(b['mel']-model.mel_mean)/model.mel_std
    fm=b['frame_mask'][...,0];pm=(b['tokens']!=0).float()
    ml=masked((m-target).abs(),fm)
    dl=masked((d-b['durations'].float().log1p()).square(),pm)
    delta=masked(((m[:,1:]-m[:,:-1])-(target[:,1:]-target[:,:-1])).abs(),fm[:,1:]*fm[:,:-1])
    def prosody_loss(pred,target,mask):
        voiced=mask*target[...,1]
        return (masked((pred[...,0]-target[...,0]).abs(),voiced)+
            masked((pred[...,2]-target[...,2]).abs(),mask)+
            masked(F.binary_cross_entropy_with_logits(pred[...,1],target[...,1],reduction='none'),mask))
    pl=prosody_loss(pr,b['prosody'],fm)
    phone=prosody_loss(p,b['phone_prosody'],pm*(b['durations']>0))
    loss=ml+.5*delta+.5*dl+.15*pl+.1*phone
    metrics=dict(total=loss.detach(),mel_normalized=ml.detach(),mel_native=masked((m-target).abs()*model.mel_std,fm).detach(),
        duration=dl.detach(),delta=delta.detach(),prosody=pl.detach(),phone_prosody=phone.detach())
    return loss,metrics


def training_objective(model,b):
    return acoustic_objective(model,b,recorded=True)
