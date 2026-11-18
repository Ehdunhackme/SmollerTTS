"""Deterministic, length-bucketed complete-utterance acoustic batches."""
import json,random
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .blocks import duration_indices


@lru_cache(maxsize=512)
def item(path):
    with np.load(path,allow_pickle=False) as f:
        x={k:torch.from_numpy(f[k].copy()) for k in ('tokens','stress','durations','mel','prosody')}
    d=x['durations'];x['indices'],x['position']=duration_indices(d)
    assert len(x['indices'])==len(x['mel'])==len(x['prosody'])
    phone=torch.zeros(len(d),3);phone.index_add_(0,x['indices'],x['prosody'])
    phone/=d.clamp_min(1)[:,None];x['phone_prosody']=phone
    return x


def batch(rows):
    xs=[item(r['features']) for r in rows];b=len(xs)
    n=max(len(x['tokens']) for x in xs);t=max(len(x['mel']) for x in xs)
    out={k:torch.zeros(b,n,dtype=torch.long) for k in ('tokens','stress','durations')}
    out.update(indices=torch.zeros(b,t,dtype=torch.long),position=torch.zeros(b,t),
        frame_mask=torch.zeros(b,t,1),mel=torch.zeros(b,t,80),prosody=torch.zeros(b,t,3),
        phone_prosody=torch.zeros(b,n,3))
    for j,x in enumerate(xs):
        nn=len(x['tokens']);tt=len(x['mel'])
        for k in ('tokens','stress','durations','phone_prosody'):out[k][j,:nn]=x[k]
        for k in ('indices','position','mel','prosody'):out[k][j,:tt]=x[k]
        out['frame_mask'][j,:tt]=1
    out['ids']=[r['id'] for r in rows]
    return out


class Steps(Dataset):
    def __init__(self,rows,start,stop,seed=2718,max_batch=24,frame_budget=12000):
        if not rows or any(r['split']!='train' for r in rows):raise ValueError('Only training rows')
        self.rows=sorted(rows,key=lambda r:(r['frames'],r['id']))
        self.start,self.stop,self.seed=start,stop,seed
        self.max_batch,self.frame_budget=max_batch,frame_budget
    def __len__(self):return self.stop-self.start
    def __getitem__(self,index):
        rng=random.Random(self.seed+(self.start+index)*1000003)
        anchor=rng.randrange(len(self.rows));lo=max(0,anchor-96);hi=min(len(self.rows),anchor+97)
        count=max(1,min(self.max_batch,self.frame_budget//self.rows[hi-1]['frames']))
        rows=[self.rows[anchor]]+[self.rows[rng.randrange(lo,hi)] for _ in range(count-1)]
        return batch(rows)


def init_worker(_):torch.set_num_threads(1)
