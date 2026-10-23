#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grade-blind Residual MLP++ (384→128→ResBlock×2→64→1) Rank DDS for SPIDER.

Only scores single MRI VAE latent features; 21-frame G-conditioned generated
trajectories supply TRAIN pair ordering. Real MRI grades used ONLY in optional
post-freeze association tables. Grade-free downstream != grade-free generator.

python residual_mlp_plusplus_dds.py train --root /path/to/SPIDER-data
python residual_mlp_plusplus_dds.py predict --model-dir results_residual_mlp_plusplus_384 --latents new.npy --manifest cases.csv
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

VERSION = 'RESIDUAL_MLP_PLUSPLUS_384_GRADE_BLIND_1.0'
LEVEL_NAMES = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']
GAPS = (3, 5, 10, 20)
STAGES = ('G1-G2', 'G2-G3', 'G3-G4', 'G4-G5')
TOLERANCES = (0., 0.25, 0.5, 1., 2.)


def fail(s):
    raise RuntimeError(str(s))

def json_write(obj, path):
    def clean(v):
        if isinstance(v, Path): return str(v)
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, float) and not math.isfinite(v): return None
        if isinstance(v, dict): return {str(k):clean(z) for k,z in v.items()}
        if isinstance(v,(list,tuple)): return [clean(z) for z in v]
        return v
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csv_write(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')

def patient_uid(sample_id, image_path):
    s = str(sample_id).strip().lower()
    if s.startswith('data1::'):
        return 'data1::' + re.sub(r'_disc\d+$', '', Path(str(image_path)).stem.lower())
    if s.startswith('data2::'):
        parts = s.split('::')
        return '::'.join(parts[:-1]) if len(parts)>2 else s
    fail('Unknown source/patient naming convention: '+s)

def blind_metadata(p):
    required = ['sample_id', 'split', 'level_index', 'image_path']
    header = pd.read_csv(p, nrows=0, encoding='utf-8-sig').columns
    if not set(required).issubset(header):fail('Metadata missing '+str(set(required)-set(header)))
    m = pd.read_csv(p, usecols=required, dtype={'sample_id':str}, encoding='utf-8-sig', low_memory=False)
    m['split'] = m['split'].astype(str).str.strip().str.lower()
    m['level_index'] = pd.to_numeric(m.level_index, errors='raise').astype(int)
    if not m.level_index.between(0,4).all():fail('level_index must be 0..4')
    if m.image_path.isna().any() or m.image_path.astype(str).duplicated().any():fail('Invalid/duplicate MRI image paths')
    m['patient_uid'] = [patient_uid(s,p) for s,p in zip(m.sample_id,m.image_path)]
    m['source'] = m.sample_id.str.split('::').str[0]
    if not set(m.split.unique()).issubset({'train','val','test'}):fail('Invalid split label')
    return m

def dev_split(split, traj_dir, cache_dir, metadata):
    cache_dir, traj_dir = Path(cache_dir), Path(traj_dir)
    files = [cache_dir/'latent_cache_meta.json', cache_dir/'source_latents.npy',
             cache_dir/'trajectory_latents.npy', traj_dir/'trajectory_metrics_all.csv']
    for p in files:
        if not p.is_file():fail('Missing cache/trajectory metadata: '+str(p))
    info = json.loads(files[0].read_text(encoding='utf-8'))
    names = info.get('case_names', [])
    real = np.load(files[1], mmap_mode='r'); traj=np.load(files[2],mmap_mode='r')
    if not names or len(set(names))!=len(names):fail(f'{split}: invalid case_names')
    if (real.ndim!=2 or traj.ndim!=3 or traj.shape[1]!=21 or
        traj.shape[2]!=real.shape[1] or len(real)!=len(names) or len(traj)!=len(names)):
        fail(f'{split}: VAE latent shapes disagree')
    if info.get('n_cases')!=len(names):fail(f'{split}: stale cache count')
    qfile = cache_dir/'q_values.npy'
    if qfile.is_file():
        q=np.load(qfile)
        if q.shape!=(21,) or not np.all(np.diff(q)>0):fail(f'{split}: invalid trajectory nodes')
    src=pd.read_csv(files[3], usecols=lambda s:s in {'step_file_path','metadata_index','sample_id','level_index'},
                    dtype={'sample_id':str}, encoding='utf-8-sig', low_memory=False)
    if 'step_file_path' not in src:fail(f'{split}: step_file_path missing')
    src['_case']=src.step_file_path.astype(str).map(lambda s:Path(s).parent.parent.name)
    counts=src.groupby('_case').size()
    bad=[s for s in names if int(counts.get(s,0))!=21]
    if bad:fail(f'{split}: cache trajectories not 21 frames, e.g. {bad[:3]}')
    frame=src.drop_duplicates('_case').set_index('_case').loc[names].reset_index()
    if 'metadata_index' in frame and frame.metadata_index.notna().all():
        ids=pd.to_numeric(frame.metadata_index,errors='raise').astype(int).to_numpy()
    else:
        parsed=frame['_case'].str.extract(r'^idx_(\d+)(?:_|$)')[0]
        if parsed.isna().any():fail(f'{split}: cannot reconstruct metadata row indices')
        ids=parsed.astype(int).to_numpy()
    if (ids<0).any() or (ids>=len(metadata)).any() or len(set(ids))!=len(ids):fail(f'{split}: invalid metadata indices')
    m=metadata.iloc[ids].copy().reset_index(drop=True)
    if not (m.split==split).all():fail(f'{split}: cache contains wrong split')
    if 'level_index' in frame and not np.array_equal(m.level_index.to_numpy(),frame.level_index.to_numpy(int)):
        fail(f'{split}: cache level index mismatch')
    if 'sample_id' in frame:
        mask=frame.sample_id.notna() & frame.sample_id.str.len().gt(0)
        if m.loc[mask,'sample_id'].tolist()!=frame.loc[mask,'sample_id'].tolist():
            fail(f'{split}: cache sample_id mismatch')
    m['case_name']=names
    print(f'[{split}] {len(m)} MRI, {m.patient_uid.nunique()} patients, D={real.shape[1]}',flush=True)
    return m,real,traj

def test_split(cache_dir,metadata,optional=False):
    c=Path(cache_dir)
    p,j=c/'test_source_latents.npy',c/'test_source_cache.json'
    if not p.exists() or not j.exists():
        if optional:return None,None
        fail('TEST frozen VAE cache missing: '+str(p))
    m=metadata.loc[metadata.split=='test'].copy().reset_index(drop=True)
    x=np.load(p,mmap_mode='r'); info=json.loads(j.read_text(encoding='utf-8'))
    if (len(x)!=len(m) or info.get('n_cases')!=len(m) or
        info.get('paths',[])!=m.image_path.astype(str).tolist()):
        fail('TEST cache paths/order mismatch')
    print(f'[test] {len(m)} MRI, {m.patient_uid.nunique()} patients',flush=True)
    return m,x

def audit_splits(a,b,c):
    for an,x,bn,y in [('train',a,'val',b),('train',a,'test',c),('val',b,'test',c)]:
        if y is None:continue
        overlap=set(x.patient_uid)&set(y.patient_uid)
        if overlap:fail(f'Patient leakage {an}/{bn}: {list(overlap)[:5]}')
    print('[audit] Patient-disjoint TRAIN/VAL/TEST: OK',flush=True)

def norm_params(real, chunk=128):
    n,d=real.shape
    sums=np.zeros(d,np.float64); squares=np.zeros(d,np.float64)
    for lo in range(0,n,chunk):
        a=np.asarray(real[lo:lo+chunk],dtype=np.float64)
        if not np.isfinite(a).all():fail('Nonfinite TRAIN VAE feature')
        sums+=a.sum(0); squares+=np.einsum('ij,ij->j',a,a)
    mu=sums/n;sd=np.sqrt(np.maximum(squares/n-mu**2,0))
    good=sd[sd>1e-7]
    if not len(good):fail('All TRAIN latent channels are constant')
    sd=np.maximum(sd,max(float(np.median(good))*.05,1e-4))
    return mu.astype(np.float32),sd.astype(np.float32)

def ztransform(a,mu,sd,features=None):
    aa=np.asarray(a,dtype=np.float32)
    if features is None:z=(aa-mu)/sd
    else:z=(aa[...,features]-mu[features])/sd[features]
    if not np.isfinite(z).all():fail('Normalized features contain NaNs')
    return np.clip(z,-8.,8.).astype(np.float32)

def selected_real(a,mu,sd,features,chunk=256):
    result=np.empty((len(a),len(features)),np.float32)
    for st in range(0,len(a),chunk):
        result[st:st+chunk]=ztransform(a[st:st+chunk],mu,sd,features)
    return result

def feature_direction(train_traj,mu,sd,max_features,chunk=16):
    """Same TRAIN-only feature-selection rule as previous three-method experiment."""
    n,d=train_traj.shape[0],train_traj.shape[2]
    acc=np.zeros(d,np.float64); zero=0
    for start in range(0,n,chunk):
        a=np.asarray(train_traj[start:start+chunk],np.float32)
        delta=ztransform(a[:,-1],mu,sd)-ztransform(a[:,0],mu,sd)
        norm=np.linalg.norm(delta,axis=1)
        good=norm>1e-7
        delta[good]/=norm[good,None];delta[~good]=0
        acc+=delta.sum(axis=0,dtype=np.float64);zero+=int((~good).sum())
    if zero>n*.05:fail('More than 5% TRAIN generated trajectories have zero net change')
    idx=np.argsort(-np.abs(acc),kind='stable')[:min(max_features,d)]
    axis=acc[idx].copy();mag=np.linalg.norm(axis)
    if mag<1e-9:fail('No coherent direction to initialize Linear Rank')
    return idx.astype(np.int64),(axis/mag).astype(np.float64),{
        'n_train_trajectories':n,'zero_endpoint_trajectories':zero,
        'selected_dim':len(idx),'endpoint_consensus_norm_all_dims':float(np.linalg.norm(acc/n))}

def make_calibration(train_scores):
    lo,hi=np.quantile(train_scores,[.05,.95])
    if hi-lo<1e-10:fail('TRAIN score distribution collapsed; 0–100 cannot be calibrated')
    lg=math.log(19.0)
    return {'origin':'TRAIN real MRI raw 5th/95th quantile (grade-blind)',
            'q05':float(lo),'q95':float(hi),'slope':float(2*lg/(hi-lo)),
            'bias':float(-lg-(2*lg/(hi-lo))*lo),'mapping':'monotone sigmoid 5->95'}

def to_dds(raw,cal):
    return 100.*expit(np.clip(cal['slope']*np.asarray(raw,dtype=np.float64)+cal['bias'],-50,50))


def log(msg):
    print(msg, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(value):
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value
    if dev.startswith('cuda') and not torch.cuda.is_available():
        fail('CUDA selected but no CUDA device is available')
    return torch.device(dev)


class ResidualBlock(nn.Module):
    """Two-layer residual adapter, feature-wise projection within hidden space."""
    def __init__(self, width=128, dropout=0.08, residual_scale=0.5):
        super().__init__()
        self.scale=float(residual_scale)
        self.f=nn.Sequential(
            nn.Linear(width,width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width,width),
        )

    def forward(self, x):
        return F.gelu(x+self.scale*self.f(x))


class ResidualRankNet(nn.Module):
    """Single-MRI scorer: 384 -> 128 -> 2 residual blocks -> 64 -> 1.

    No frame-index, true grade, source or patient identity is passed in.
    Train only with order relationships in 21-frame generated trajectories.
    Input 384-dimensional subset is selected with TRAIN trajectories only.
    """
    def __init__(self, input_dim=384, width=128, head_width=64, n_blocks=2,
                 dropout=0.08, residual_scale=0.5):
        super().__init__()
        self.input_dim=int(input_dim)
        self.width=int(width)
        self.head_width=int(head_width)
        self.n_blocks=int(n_blocks)
        self.dropout=float(dropout)
        self.residual_scale=float(residual_scale)
        if self.input_dim <= 0 or self.width <= 0 or self.head_width <= 0 or self.n_blocks != 2:
            raise ValueError('Residual MLP++ requires positive dimensions and exactly 2 residual blocks')
        self.stem=nn.Sequential(
            nn.Linear(input_dim,width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.blocks=nn.Sequential(*[
            ResidualBlock(width,dropout,residual_scale)
            for _ in range(n_blocks)
        ])
        self.head=nn.Sequential(
            nn.Linear(width,head_width),
            nn.GELU(),
            nn.Linear(head_width,1),
        )

    def forward(self,z):
        h=self.stem(z)
        h=self.blocks(h)
        return self.head(h).squeeze(-1)



def long_indices(n_cases, seed, reps_per_gap, chunk=24):
    """Exact sampler order of RankCapacity/LinearRank (chunk->gap->case).
    All 20 adjacent intervals are ALWAYS used, no adjacent downsampling.
    """
    rng=np.random.default_rng(seed)
    result={g:np.empty((n_cases,min(reps_per_gap,21-g)),np.int32) for g in GAPS}
    for st in range(0,n_cases,chunk):
        hi=min(st+chunk,n_cases)
        for g in GAPS:
            k=min(reps_per_gap,21-g)
            result[g][st:hi]=np.stack([rng.choice(21-g,size=k,replace=False) for _ in range(hi-st)])
    return result


def predict_raw(net, matrix, device, batch=4096):
    net.eval()
    output=np.empty(len(matrix),np.float64)
    with torch.no_grad():
        for st in range(0,len(matrix),batch):
            t=torch.from_numpy(np.asarray(matrix[st:st+batch],np.float32)).to(device)
            output[st:st+batch]=net(t).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(output).all():
        fail('Non-finite scores: check VAE cache and model parameters')
    return output


def trajectory_raw(net, latent_memmap, mu, sd, feats, device, batch=32):
    result=np.empty((len(latent_memmap),21),np.float64)
    for st in range(0,len(latent_memmap),batch):
        hi=min(st+batch,len(latent_memmap))
        data=ztransform(np.asarray(latent_memmap[st:hi]),mu,sd,feats)
        result[st:hi]=predict_raw(net,data.reshape(-1,len(feats)),device).reshape(hi-st,21)
    return result


def val_metrics(raw):
    d=np.diff(np.asarray(raw,np.float64),axis=1)
    return {
        'complete_strict_pct':float(np.mean(np.all(d>=0,axis=1))*100),
        'adjacent_reverse_pct':float(np.mean(d<0)*100),
        'positive_endpoint_pct':float(np.mean(raw[:,-1]>raw[:,0])*100),
        'endpoint_median_raw':float(np.median(raw[:,-1]-raw[:,0]))
    }


def fit_model(train_traj, val_traj, mu, sd, feats, args, device, out):
    """Select checkpoint by G-blind VAL synthetic order, never by true grade.

    All 20 adjacent differences within every trajectory enter each epoch.
    Four long-gap groups use the same fixed TRAIN case pair indices for all seeds.
    No frame-index features or synthetic q values are fed to the model.
    """
    n=len(train_traj)
    longpos=long_indices(n,args.pair_seed,args.long_per_gap,args.chunk)
    long_counts={str(g):int(n*longpos[g].shape[1]) for g in GAPS}
    log(f'[pairs] ALL adjacent: {n*20:,}; long: {long_counts}')
    v=np.empty((len(val_traj),21,len(feats)),np.float32)
    for st in range(0,len(val_traj),32):
        hi=min(st+32,len(val_traj))
        v[st:hi]=ztransform(np.asarray(val_traj[st:hi]),mu,sd,feats)
    # Keep VAL features on CPU; stream to GPU to avoid peaks on smaller cards.
    v_flat=v.reshape(-1,len(feats))
    history=[]
    summary=[]
    torch.set_num_threads(min(torch.get_num_threads(),8))
    for seed in args.seeds:
        seed_all(int(seed))
        net=ResidualRankNet(len(feats),args.width,args.head_width,2,args.dropout,args.residual_scale).to(device)
        if seed == args.seeds[0]:
            log(f'[architecture] {len(feats)} -> {args.width} -> ResidualBlock×2 -> {args.head_width} -> 1; params={sum(p.numel() for p in net.parameters()):,}')
        opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        rg=np.random.default_rng(int(seed))
        chosen=None;best_key=None;best_ep=-1;wait=0
        for epoch in range(1,args.epochs+1):
            net.train()
            ordering=rg.permutation(n)
            weighted_loss=0.;num_seen=0
            for st in range(0,n,args.batch_size):
                ids=ordering[st:st+args.batch_size]
                x=ztransform(np.asarray(train_traj[ids]),mu,sd,feats)
                scores=net(torch.from_numpy(x).to(device).reshape(-1,len(feats))).reshape(len(ids),21)
                dif=scores[:,1:]-scores[:,:-1]
                loss=F.softplus(-dif/args.temperature).mean()
                if args.long_weight>0:
                    for g in GAPS:
                        ix=torch.from_numpy(longpos[g][ids].astype(np.int64)).to(device)
                        step=scores.gather(1,ix+g)-scores.gather(1,ix)
                        loss=loss+args.long_weight/len(GAPS)*F.softplus(-step/args.temperature).mean()
                if not torch.isfinite(loss):
                    fail(f'Nonfinite training loss, seed={seed}, epoch={epoch}')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),args.grad_clip)
                opt.step()
                weighted_loss+=float(loss.detach().cpu())*len(ids);num_seen+=len(ids)
            val_raw=predict_raw(net,v_flat,device).reshape(len(val_traj),21)
            qc=val_metrics(val_raw)
            train_loss=weighted_loss/max(num_seen,1)
            # Predeclared label-blind rule; not using VAL true grades or TEST.
            key=(round(qc['complete_strict_pct'],10), -round(qc['adjacent_reverse_pct'],10))
            row={'seed':int(seed),'epoch':epoch,'TRAIN_rank_loss':train_loss,**qc}
            history.append(row)
            if best_key is None or key>best_key:
                best_key=key
                chosen={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
                best_ep=epoch;wait=0
            else:
                wait+=1
            if epoch==1 or epoch%5==0 or wait==0:
                log(f'[seed {seed} epoch {epoch}] TRAIN loss={train_loss:.5f} VAL full={qc["complete_strict_pct"]:.2f}% reverse={qc["adjacent_reverse_pct"]:.2f}%')
            if wait>=args.patience:
                break
        cp=out/f'model_seed_{seed}.pt'
        torch.save({'state_dict':chosen,'input_dim':len(feats),'width':args.width,
             'head_width':args.head_width,'n_blocks':2,
             'dropout':args.dropout,'residual_scale':args.residual_scale,
             'seed':int(seed),'best_epoch':best_ep},cp)
        selected={r['epoch']:r for r in history if r['seed']==int(seed)}[best_ep]
        summary.append({'seed':int(seed),'selected_epoch':int(best_ep),
             'VAL_complete_strict_pct':selected['complete_strict_pct'],
             'VAL_adjacent_reverse_pct':selected['adjacent_reverse_pct'],
             'VAL_positive_endpoint_pct':selected['positive_endpoint_pct'],
             'checkpoint':str(cp)})
    csv_write(pd.DataFrame(history),out/'01_training_history.csv')
    seeds=pd.DataFrame(summary)
    csv_write(seeds,out/'02_seed_selection.csv')
    best=seeds.sort_values(['VAL_complete_strict_pct','VAL_adjacent_reverse_pct','seed'],ascending=[False,True,True]).iloc[0]
    winner_path=Path(str(best.checkpoint))
    payload=torch.load(winner_path,map_location='cpu',weights_only=False)
    torch.save(payload,out/'model_residual_mlp_plusplus.pt')
    final=ResidualRankNet(len(feats),args.width,args.head_width,2,args.dropout,args.residual_scale).to(device)
    final.load_state_dict(payload['state_dict']);final.eval()
    details={
        'version':VERSION,'model':'ResidualMLP++',
        'architecture':f'{len(feats)} -> {args.width} -> 2 x ResidualBlock({args.width}) -> {args.head_width} -> 1',
        'trainable_parameters':int(sum(p.numel() for p in final.parameters())),
        'selection_rule':'max VAL generated complete STRICT monotonicity, then min VAL reverse rate, then lowest seed',
        'selected_seed':int(best.seed),'selected_epoch':int(best.selected_epoch),
        'selected_VAL_complete_strict_pct':float(best.VAL_complete_strict_pct),
        'selected_VAL_adjacent_reverse_pct':float(best.VAL_adjacent_reverse_pct),
        'n_train_adjacent_pairs':int(n*20), 'long_pairs':long_counts,
        'true_G_used_in_training':False,'true_G_used_for_VAL_checkpoint_selection':False,
        'generator_originally_G_conditioned':True,
        'TEST_used_for_checkpoint_selection':False,
        'all_seeds':summary}
    json_write(details,out/'BEST_MODEL.json')
    return final,details


def trajectory_detail(raw,cal,frame,split):
    raw=np.asarray(raw,np.float64)
    d_raw=np.diff(raw,axis=1)
    scored=to_dds(raw,cal)
    d_dds=np.diff(scored,axis=1)
    reverse=d_raw<0
    by_group=[];by_stage=[];by_position=[];by_case=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        rec={'split':split,'case_name':str(row.case_name),
             'patient_uid':str(row.patient_uid),'source':str(row.source),
             'level_index':int(row.level_index),
             'DDS_start':float(scored[i,0]),'DDS_end':float(scored[i,-1]),
             'net_DDS':float(scored[i,-1]-scored[i,0]),
             'reverse_count':int(reverse[i].sum()),
             'largest_reverse_DDS':float(np.max(np.where(reverse[i],-d_dds[i],0.))),
             'total_reverse_DDS':float(np.sum(np.where(reverse[i],-d_dds[i],0.)))}
        for tol in TOLERANCES:
            rec[f'complete_tol_{tol:g}']=int(np.all(~reverse[i]) if tol==0 else np.all(d_dds[i]>=-tol))
        by_case.append(rec)
    for lvl in [-1,0,1,2,3,4]:
        for source in ['ALL']+sorted(frame.source.unique().tolist()):
            mask=np.ones(len(frame),bool) if lvl<0 else frame.level_index.to_numpy()==lvl
            if source!='ALL':mask &= frame.source.to_numpy()==source
            if not mask.any():continue
            rr=reverse[mask];dd=d_dds[mask];back=-dd[rr]
            row={'split':split,'level':'ALL' if lvl<0 else LEVEL_NAMES[lvl],
                 'source':source,'n_trajectories':int(mask.sum()),
                 'adjacent_reverse_pct':float(rr.mean()*100),
                 'positive_endpoint_pct':float(np.mean(raw[mask,-1]>raw[mask,0])*100),
                 'median_reverse_DDS':float(np.median(back)) if len(back) else 0.,
                 'p95_reverse_DDS':float(np.percentile(back,95)) if len(back) else 0.}
            for tol in TOLERANCES:
                row[f'full_monotone_tol_{tol:g}_pct']=float(np.mean(np.all(~rr,axis=1) if tol==0 else np.all(dd>=-tol,axis=1))*100)
                row[f'adjacent_reverse_tol_{tol:g}_pct']=float(np.mean(rr if tol==0 else dd < -tol)*100)
            by_group.append(row)
    for stage in range(4):
        lo,hi=5*stage,5*(stage+1)
        rev=reverse[:,lo:hi]
        negative=(-d_dds[:,lo:hi])[rev]
        by_stage.append({'split':split,'stage':STAGES[stage],
                 'n_pairs':int(rev.size),'reverse_pct':float(rev.mean()*100),
                 'median_reverse_DDS':float(np.median(negative)) if len(negative) else 0.,
                 'p95_reverse_DDS':float(np.percentile(negative,95)) if len(negative) else 0.})
    for pos in range(20):
        rev=reverse[:,pos];neg=(-d_dds[:,pos])[rev]
        by_position.append({'split':split,'position':pos,'stage':STAGES[pos//5],
                 'reverse_pct':float(rev.mean()*100),
                 'p95_reverse_DDS':float(np.percentile(neg,95)) if len(neg) else 0.})
    return map(pd.DataFrame,(by_group,by_stage,by_position,by_case))


def get_real_predictions(net,frame,matrix,mu,sd,features,cal,device):
    z=selected_real(matrix,mu,sd,features)
    s=predict_raw(net,z,device)
    out=frame[['image_path','sample_id','patient_uid','source','level_index','split']].copy()
    out['raw_score']=s
    out['DDS']=to_dds(s,cal)
    return out


def real_distribution(pred):
    rows=[]
    for (split,source),grp in pred.groupby(['split','source']):
        for lvl in [-1,0,1,2,3,4]:
            sub=grp if lvl<0 else grp.loc[grp.level_index==lvl]
            if not len(sub):continue
            a=sub.DDS.to_numpy(float)
            rows.append({'split':split,'source':source,'level':'ALL' if lvl<0 else LEVEL_NAMES[lvl],
                         'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
                         'q05':float(np.quantile(a,.05)),'q25':float(np.quantile(a,.25)),
                         'q75':float(np.quantile(a,.75)),'q95':float(np.quantile(a,.95)),
                         'DDS_le1_pct':float(np.mean(a<=1)*100),
                         'DDS_ge99_pct':float(np.mean(a>=99)*100)})
    return pd.DataFrame(rows)


def external_grade_metrics(meta_path,preds,out):
    # Function invoked ONLY AFTER checkpoint, features, calibration, blind metrics are frozen.
    grades=pd.read_csv(meta_path,usecols=['image_path','grade'],encoding='utf-8-sig')
    if grades.image_path.duplicated().any():fail('Duplicated external image paths')
    grades['grade']=pd.to_numeric(grades.grade,errors='raise').astype(int)
    merged=preds.merge(grades,on='image_path',validate='many_to_one',how='left')
    if merged.grade.isna().any():fail('External grade association failed for some MRI paths')
    rows=[]
    for split,group in merged.groupby('split'):
        for lvl in [-1,0,1,2,3,4]:
            part=group if lvl<0 else group[group.level_index==lvl]
            if len(part)<10 or part.grade.nunique()<2:continue
            rho=float(spearmanr(part.DDS,part.grade).statistic)
            agree=0.;total=0
            for low in range(1,6):
                for high in range(low+1,6):
                    left=part.loc[part.grade==low,'DDS'].to_numpy(float)
                    right=part.loc[part.grade==high,'DDS'].to_numpy(float)
                    if not len(left) or not len(right):continue
                    d=right[:,None]-left[None,:]
                    agree+=float(np.sum(d>0)+.5*np.sum(d==0));total+=d.size
            rows.append({'split':split,'level':'ALL' if lvl<0 else LEVEL_NAMES[lvl],
              'n':len(part),'spearman_G_POST_HOC':rho,
              'cross_grade_concordance_POST_HOC':agree/total if total else float('nan')})
    csv_write(pd.DataFrame(rows),out/'09_EXTERNAL_G_POST_HOC.csv')
    return pd.DataFrame(rows)


def patient_bootstrap_val(cases,rep,seed):
    sub=cases[cases.split=='val']
    patients=sub.patient_uid.astype(str).to_numpy()
    groups=[np.flatnonzero(patients==p) for p in np.unique(patients)]
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(rep):
        ids=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        df=sub.iloc[ids]
        values.append([float(df.complete_tol_0.mean()*100),float(df.complete_tol_1.mean()*100)])
    a=np.asarray(values,float)
    return {'n_bootstrap':rep,'n_patients':len(groups),
            'full_strict_ci95':np.percentile(a[:,0],[2.5,97.5]).tolist(),
            'full_tol1_ci95':np.percentile(a[:,1],[2.5,97.5]).tolist()}


def draw_figures(summary,stages,pred,out):
    val=summary[(summary.split=='val')&(summary.level=='ALL')&(summary.source=='ALL')]
    if len(val):
        r=val.iloc[0]
        fig,ax=plt.subplots(figsize=(9,5))
        t=np.asarray(TOLERANCES)
        vals=[r[f'full_monotone_tol_{v:g}_pct'] for v in t]
        ax.plot(range(len(t)),vals,marker='o',linewidth=2)
        ax.set_xticks(range(len(t)),[str(v) for v in t]);ax.set_ylim(0,105)
        ax.set_xlabel('Allowed reverse step (DDS points)');ax.set_ylabel('VAL fully monotone trajectories (%)')
        ax.set_title('Residual MLP++: strict vs tolerant monotonicity')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'Fig1_VAL_tolerance.png',dpi=190);plt.close(fig)
    ss=stages[stages.split=='val']
    if len(ss):
        fig,ax=plt.subplots(figsize=(9,5))
        ax.bar(ss.stage,ss.reverse_pct)
        ax.set_ylabel('VAL adjacent reverse steps (%)');ax.set_title('VAL reverse steps by generated stage')
        ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'Fig2_VAL_stage_reverse.png',dpi=190);plt.close(fig)
    p=pred[pred.split=='val']
    if len(p):
        fig,ax=plt.subplots(figsize=(9,5))
        ax.hist(p.DDS,bins=np.arange(0,105,5),density=True)
        ax.set_xlabel('Grade-blind DDS');ax.set_ylabel('VAL real MRI density')
        ax.set_title('Residual MLP++: frozen real-MRI DDS')
        fig.tight_layout();fig.savefig(out/'Fig3_VAL_DDS_distribution.png',dpi=190);plt.close(fig)


def load_model(model_dir,device):
    folder=Path(model_dir)
    features=np.load(folder/'feature_normalizer.npz')
    mu=features['mean'];sd=features['std'];indices=features['selected']
    payload=torch.load(folder/'model_residual_mlp_plusplus.pt',map_location='cpu',weights_only=False)
    if payload.get('n_blocks') != 2 or payload.get('input_dim') != len(indices):
        fail('Saved model architecture and feature_normalizer do not match')
    model=ResidualRankNet(payload['input_dim'],payload['width'],payload['head_width'],payload['n_blocks'],
                          payload['dropout'],payload['residual_scale']).to(device)
    model.load_state_dict(payload['state_dict']);model.eval()
    calibration=json.loads((folder/'calibration.json').read_text(encoding='utf8'))
    return model,mu,sd,indices,calibration


def train(args):
    started=time.time()
    seed_all(args.pair_seed)
    device=select_device(args.device)
    root=Path(args.root).expanduser().resolve()
    out=Path(args.out or root/'results_residual_mlp_plusplus_384').expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    meta_file=Path(args.metadata_csv or root/'data/prepared_metadata.csv')
    cache=Path(args.cache_root or root/'results_lcovt_dds_final')
    metadata=blind_metadata(meta_file)  # Only sample_id, split, level_index, image_path.
    tr,Xtr,Ttr=dev_split('train',args.train_trajectory_dir or root/'results_lcovt_G1_G5_21frames_train_newvae_batch21',cache/'cache_train',metadata)
    va,Xva,Tva=dev_split('val',args.val_trajectory_dir or root/'results_lcovt_G1_G5_21frames_val_newvae_batch21',cache/'cache_val',metadata)
    te,Xte=(None,None) if args.skip_test else test_split(cache/'cache_test_source_only',metadata,optional=args.allow_missing_test)
    audit_splits(tr,va,te)
    if Xtr.shape[1]!=Xva.shape[1] or (Xte is not None and Xtr.shape[1]!=Xte.shape[1]):
        fail('Different VAE latent dimensions in splits')
    if args.features<=0 or args.width<=0 or args.head_width<=0 or args.patience<=0 or args.batch_size<=0 or args.epochs<=0 or args.long_per_gap<=0:
        fail('Positive --features, --width, --head-width, --patience, --batch-size, --epochs and --long-per-gap required')
    if not (args.temperature>0 and args.long_weight>=0 and args.lr>0):
        fail('Invalid optimizer hyperparameters')
    log(f'[{VERSION}] ONLY Residual MLP++ | device={device} | real G BLIND')
    mean,std=norm_params(Xtr)
    selected,unused_axis,selection_qc=feature_direction(Ttr,mean,std,args.features,args.chunk)
    np.savez_compressed(out/'feature_normalizer.npz',mean=mean,std=std,selected=selected)
    json_write(selection_qc,out/'00_feature_selection_QC.json')
    log(f'[features] {len(selected)}/{len(mean)} selected from TRAIN generated endpoints')
    net,info=fit_model(Ttr,Tva,mean,std,selected,args,device,out)
    # Calibration uses real TRAIN latent scores but NO grades.
    train_raw=predict_raw(net,selected_real(Xtr,mean,std,selected),device)
    calibration=make_calibration(train_raw)
    json_write(calibration,out/'calibration.json')
    splits=[('train',tr,Xtr),('val',va,Xva)]
    if te is not None:splits.append(('test',te,Xte))
    pred=pd.concat([get_real_predictions(net,frame,x,mean,std,selected,calibration,device)
                    for _,frame,x in splits],ignore_index=True)
    csv_write(pred,out/'08_predictions_all_real_BLIND.csv')
    if te is not None:
        csv_write(pred[pred.split=='test'],out/'predictions_test_residual_mlp_plusplus.csv')
    groups=[];stages=[];positions=[];cases=[]
    for split,frame,t in [('train',tr,Ttr),('val',va,Tva)]:
        log(f'[QC] complete {split} generated trajectories: {len(frame)}')
        raw=trajectory_raw(net,t,mean,std,selected,device,args.chunk)
        gg,ss,pp,cc=trajectory_detail(raw,calibration,frame,split)
        groups.append(gg);stages.append(ss);positions.append(pp);cases.append(cc)
    gg=pd.concat(groups,ignore_index=True)
    ss=pd.concat(stages,ignore_index=True)
    pp=pd.concat(positions,ignore_index=True)
    cc=pd.concat(cases,ignore_index=True)
    csv_write(gg,out/'03_trajectory_QC_by_level_source_BLIND.csv')
    csv_write(ss,out/'04_trajectory_backward_by_stage_BLIND.csv')
    csv_write(pp,out/'05_trajectory_backward_by_position_BLIND.csv')
    csv_write(cc,out/'06_trajectory_cases_BLIND.csv')
    csv_write(real_distribution(pred),out/'07_real_DDS_distribution_by_source_level_BLIND.csv')
    if args.bootstrap>0:
        json_write(patient_bootstrap_val(cc,args.bootstrap,args.pair_seed),out/'10_VAL_patient_bootstrap_CI_BLIND.json')
    draw_figures(gg,ss,pred,out)
    # Evidence that everything is frozen before optional real-G external report.
    json_write({'version':VERSION,'model_frozen':True,'selected_seed':info['selected_seed'],
                'true_G_used_to_fit_model':False,'true_G_used_for_checkpoint_choice':False,
                'true_G_used_for_calibration':False,'TEST_used_for_selection':False,
                'generator_previously_G_conditioned':True},out/'MODEL_FROZEN_before_G_evaluation.json')
    external=None
    if not args.no_grade_eval:
        log('[evaluation-only] checkpoint/calibration FROZEN, now read real Pfirrmann grades')
        external=external_grade_metrics(meta_file,pred,out)
    tr_q=gg[(gg.split=='train')&(gg.level=='ALL')&(gg.source=='ALL')].iloc[0]
    va_q=gg[(gg.split=='val')&(gg.level=='ALL')&(gg.source=='ALL')].iloc[0]
    g4=ss[(ss.split=='val')&(ss.stage=='G4-G5')].iloc[0]
    lines=['# Residual MLP++ Rank DDS — single model','',
      f'- Version: {VERSION}',
      '- Frozen VAE; source 21-frame generator previously G-conditioned.',
      '- TRAIN real Pfirrmann labels NEVER entered training, feature selection, checkpoint choice or DDS calibration.',
      '- TRAIN includes every adjacent (20 per case) generated pair; long gaps 3,5,10,20.',
      '- Checkpoint chosen using VAL generated ordering, not real G labels or TEST.',
      f'- Model architecture: {len(selected)} → {args.width} → residual blocks ×2 ({args.width}→{args.width}) → {args.head_width} → 1.',
      f'- Seeds: {args.seeds}; selected seed: {info["selected_seed"]}, epoch: {info["selected_epoch"]}.','',
      '## Generated 21-frame trajectory diagnostics','',
      '| Split | Full strict monotone | Adjacent reverse | Full tolerance 1 DDS |','|---|---:|---:|---:|',
      f'| TRAIN | {tr_q.full_monotone_tol_0_pct:.2f}% | {tr_q.adjacent_reverse_pct:.2f}% | {tr_q.full_monotone_tol_1_pct:.2f}% |',
      f'| VAL | {va_q.full_monotone_tol_0_pct:.2f}% | {va_q.adjacent_reverse_pct:.2f}% | {va_q.full_monotone_tol_1_pct:.2f}% |',
      f'\n- VAL G4→G5 reverse: {g4.reverse_pct:.2f}%.',
      '\n## External real grade association (post-freeze only)','']
    if external is not None:
        for split in ['val','test']:
            part=external[(external.split==split)&(external.level=='ALL')]
            if len(part):lines.append(f'- {split.upper()} Spearman vs Pfirrmann: {part.iloc[0].spearman_G_POST_HOC:.4f} (POST-HOC only).')
    else:lines.append('- Real Pfirrmann grades not read (--no-grade-eval).')
    lines.extend(['', '## Cautions',
      '- Synthetic trajectory monotonicity is NOT independent medical severity validation.',
      '- TEST has been previously explored; do not tune architecture/hyperparameters against TEST.',
      '- Real MRI within-grade severity needs clinician or independent phenotype validation.',
      '- DDS 0–100 is a TRAIN-anchored relative score, not biological percentage.',
      f'- Wall time: {time.time()-started:.1f}s',''])
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    log('\n'.join(lines));log(f'[DONE] {out}')


def predict(args):
    device=select_device(args.device)
    model,mu,sd,feats,cal=load_model(args.model_dir,device)
    raw=np.load(args.latents,mmap_mode='r')
    manifest=pd.read_csv(args.manifest,encoding='utf-8-sig')
    if raw.ndim!=2 or raw.shape[1]!=len(mu) or len(raw)!=len(manifest):
        fail(f'Latents must be [N,{len(mu)}] and match manifest row count')
    if 'image_path' not in manifest:
        fail('Manifest needs image_path column')
    z=selected_real(raw,mu,sd,feats)
    scores=predict_raw(model,z,device)
    for col in ['grade','true_grade','DDS','raw_score']:
        if col in manifest:manifest=manifest.drop(columns=[col])
    manifest['raw_score']=scores
    manifest['DDS']=to_dds(scores,cal)
    outfile=Path(args.output or Path(args.model_dir)/'predictions_new_residual_mlp_plusplus.csv')
    outfile.parent.mkdir(parents=True,exist_ok=True)
    csv_write(manifest,outfile)
    log(f'[PREDICT] {len(scores)} images -> {outfile}')


def build_parser():
    p=argparse.ArgumentParser(description='SPIDER single-model Grade-blind Residual MLP++ 384-dimensional continuous DDS')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('train',help='fit only Residual MLP and generate experiment tables')
    a.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a.add_argument('--out')
    a.add_argument('--metadata-csv');a.add_argument('--cache-root')
    a.add_argument('--train-trajectory-dir');a.add_argument('--val-trajectory-dir')
    a.add_argument('--skip-test',action='store_true');a.add_argument('--allow-missing-test',action='store_true')
    a.add_argument('--no-grade-eval',action='store_true')
    a.add_argument('--features',type=int,default=384);a.add_argument('--width',type=int,default=128)
    a.add_argument('--head-width',type=int,default=64)
    a.add_argument('--dropout',type=float,default=0.08);a.add_argument('--residual-scale',type=float,default=0.5)
    a.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028,2029,2030])
    a.add_argument('--pair-seed',type=int,default=2026)
    a.add_argument('--device',default='auto');a.add_argument('--chunk',type=int,default=24)
    a.add_argument('--batch-size',type=int,default=128)
    a.add_argument('--epochs',type=int,default=80);a.add_argument('--patience',type=int,default=32)
    a.add_argument('--lr',type=float,default=5e-4);a.add_argument('--weight-decay',type=float,default=1e-3)
    a.add_argument('--temperature',type=float,default=0.2)
    a.add_argument('--long-weight',type=float,default=0.35)
    a.add_argument('--long-per-gap',type=int,default=3)
    a.add_argument('--grad-clip',type=float,default=5.0)
    a.add_argument('--bootstrap',type=int,default=300)
    b=s.add_parser('predict',help='predict independent MRI using PRE-ENCODED VAE latent matrix')
    b.add_argument('--model-dir',required=True);b.add_argument('--latents',required=True)
    b.add_argument('--manifest',required=True);b.add_argument('--output')
    b.add_argument('--device',default='cpu')
    return p


if __name__=='__main__':
    args=build_parser().parse_args()
    if args.command=='train':train(args)
    else:predict(args)


