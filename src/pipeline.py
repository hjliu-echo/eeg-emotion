"""Evaluation revision prepared with AI assistance in November 2026.

This is new curation code, not a claim about the original course submission.
Only synthetic runs have been verified unless real-data results are supplied.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import copy
import platform

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from electrodes import CH_NAMES, coords

MAP3 = np.array([0, 0, 0, 1, 2, 2, 2])


def prepare(data_root, output, band, classes):
    """Read provider DE derivatives, keeping trial and subject identities."""
    xs, ys, subjects, contexts, trials, starts, sources = [], [], [], [], [], [], []
    seen = set()
    for path in sorted(Path(data_root).rglob('*_task-emotion_de.npy')):
        m = re.fullmatch(r'sub-(\d+)_ses-(ima|vid)_task-emotion_de.npy', path.name)
        if not m:
            continue
        sid, context = int(m[1]), m[2]
        if not 1 <= sid <= 54:
            continue  # fixed trial labels are only safe for complete records
        if (sid, context) in seen:
            raise ValueError(f'Duplicate subject/session: {sid}, {context}')
        seen.add((sid, context))
        de = np.load(path, allow_pickle=False)
        if de.shape == (64, 630, 5):
            de = de.reshape(64, 21, 30, 5)
        if de.shape != (64, 21, 30, 5) or not np.isfinite(de).all():
            raise ValueError(f'Unsupported or nonfinite DE array: {path.name}, {de.shape}')
        sources.append({'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for t in range(21):
            for start in range(0, 28, 4):
                grid = np.zeros((9, 9, 4), dtype=np.float32)
                for i, ch in enumerate(CH_NAMES):
                    r, c = coords[ch]
                    grid[r,c] = de[i,t,start:start+4,band]
                interp = RegularGridInterpolator((np.arange(9), np.arange(9)), grid, method='cubic')
                yy, xx = np.mgrid[0:8:32j,0:8:32j]
                x = interp(np.stack([yy,xx],-1)).transpose(2,0,1).astype(np.float32)
                xs.append(x); ys.append(MAP3[t//3] if classes == 3 else t//3)
                subjects.append(sid); contexts.append(context); trials.append(t); starts.append(start)
    if not xs:
        raise ValueError('No complete-subject DE derivatives found. Download the provider derivatives first.')
    output=Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.stack(xs), y=ys, subject=subjects, context=contexts,
                        trial=trials, start=starts, classes=classes, band=band,
                        synthetic=False)
    output.with_suffix('.sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')


def split_indices(z, protocol, subject, context, fold, seed):
    """Validation never contains the test subject or held-out stimulus groups."""
    subject_ids=np.asarray(z['subject']); contexts=np.asarray(z['context'])
    trial=np.asarray(z['trial']); y=np.asarray(z['y'])
    if protocol == 'loso':
        use=np.arange(len(y)) if context=='multi' else np.flatnonzero(contexts==context)
        test=use[subject_ids[use]==subject]
        development=use[subject_ids[use]!=subject]
        if len(np.unique(subject_ids[development])) < 2:
            raise ValueError('LOSO requires at least two development subjects and one test subject.')
        val_sid=np.unique(subject_ids[development])[seed % len(np.unique(subject_ids[development]))]
        val=development[subject_ids[development]==val_sid]
        train=development[subject_ids[development]!=val_sid]
    else:
        use=np.flatnonzero((subject_ids==subject) & ((contexts==context) if context!='multi' else True))
        if not len(use):
            raise ValueError('No samples for selected subject/context.')
        # Group the corresponding stimulus index across both contexts too.
        groups=np.unique(trial[use])
        group_labels=[]
        for group in groups:
            labels=np.unique(y[use[trial[use]==group]])
            if len(labels)!=1:raise ValueError('A stimulus group has inconsistent labels.')
            group_labels.append(labels[0])
        group_labels=np.asarray(group_labels)
        outer=StratifiedKFold(3,shuffle=True,random_state=seed)
        dev_group_idx,test_group_idx=list(outer.split(groups,group_labels))[fold]
        inner=StratifiedKFold(2,shuffle=True,random_state=seed+1)
        tr_group_idx,va_group_idx=next(inner.split(groups[dev_group_idx],group_labels[dev_group_idx]))
        train=use[np.isin(trial[use],groups[dev_group_idx[tr_group_idx]])]
        val=use[np.isin(trial[use],groups[dev_group_idx[va_group_idx]])]
        test=use[np.isin(trial[use],groups[test_group_idx])]
    if any(len(i)==0 for i in [train,val,test]):
        raise ValueError('Empty split.')
    expected=set(range(int(z['classes'])))
    for idx in [train,val,test]:
        if set(y[idx])!=expected:
            raise ValueError('A split is missing classes; revise group allocation before training.')
    groups=subject_ids if protocol=='loso' else trial
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not set(a)&set(b)
        assert not set(groups[a])&set(groups[b])
    return train,val,test


def fit_stats(x, train):
    mean=x[train].mean(axis=(0,2,3),keepdims=True)
    std=x[train].std(axis=(0,2,3),keepdims=True)
    return mean,np.maximum(std,1e-6)


def build_swin(classes):
    sys.path.insert(0,str(HERE.parent/'third_party/st_shap'))
    from SWIN_trans import SwinTransformer
    return SwinTransformer(img_size=32,patch_size=4,in_chans=4,num_classes=classes,
        embed_dim=96,depths=[2,2,6,2],num_heads=[3,6,12,24],window_size=2,
        mlp_ratio=4.,qkv_bias=True,drop_rate=.17,attn_drop_rate=.17,
        drop_path_rate=.1,patch_norm=True,use_checkpoint=False)


def run(args):
    random.seed(args.seed);np.random.seed(args.seed)
    with np.load(args.cache,allow_pickle=False) as data:
        z={k:data[k] for k in data.files}
    train,val,test=split_indices(z,args.protocol,args.subject,args.context,args.fold,args.seed)
    x,y=z['x'],z['y']
    if x.ndim!=4 or x.shape[1:]!=(4,32,32) or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 4 x 32 x 32 input.')
    mean,std=fit_stats(x,train)
    versions={'python':platform.python_version(),'numpy':np.__version__}
    if args.model in ['logreg','majority']:
        xx=x.reshape(len(x),-1)
        estimator=(make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=args.seed))
                   if args.model=='logreg' else DummyClassifier(strategy='most_frequent'))
        estimator.fit(xx[train],y[train])
        pred=estimator.predict(xx[test])
        best_epoch=None
    else:
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        net=build_swin(int(z['classes'])).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-3)
        xx=((x-mean)/std).astype(np.float32)
        def loader(idx,shuffle=False):
            return DataLoader(TensorDataset(torch.from_numpy(xx[idx]),torch.from_numpy(y[idx]).long()),
                              batch_size=args.batch_size,shuffle=shuffle)
        @torch.no_grad()
        def predict(idx):
            net.eval()
            return np.concatenate([net(a.to(device)).argmax(1).cpu().numpy() for a,_ in loader(idx)])
        best=-np.inf;best_state=None;best_epoch=None
        for epoch in range(args.epochs):
            net.train()
            for a,b in loader(train,True):
                opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(a.to(device)),b.to(device))
                loss.backward();opt.step()
            score=balanced_accuracy_score(y[val],predict(val))
            if score>best:
                best=score;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        net.load_state_dict(best_state)
        pred=predict(test)  # evaluate once after validation-based selection
        versions['torch']=torch.__version__
    import sklearn,scipy
    versions.update(sklearn=sklearn.__version__,scipy=scipy.__version__)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)  # refuse to overwrite prior evidence
    summary={'data_status':'synthetic_check_only' if bool(z.get('synthetic',False)) else 'real_data',
       'accuracy':accuracy_score(y[test],pred),'balanced_accuracy':balanced_accuracy_score(y[test],pred),
       'macro_f1':f1_score(y[test],pred,labels=list(range(int(z['classes']))),average='macro',zero_division=0),
       'confusion_matrix':confusion_matrix(y[test],pred,labels=list(range(int(z['classes'])))).tolist(),
       'counts':{'train':len(train),'validation':len(val),'test':len(test)},
       'best_epoch':best_epoch,'configuration':vars(args),'versions':versions,
       'cache_sha256':hashlib.sha256(Path(args.cache).read_bytes()).hexdigest()}
    (output/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'split_indices.json').write_text(json.dumps({k:v.tolist() for k,v in
        [('train',train),('validation',val),('test',test)]},indent=2),encoding='utf-8')
    if args.model=='logreg':
        scaler=estimator.named_steps['standardscaler']
        np.savez_compressed(output/'train_stats.npz',mean=scaler.mean_,std=scaler.scale_)
    elif args.model=='swin':
        np.savez_compressed(output/'train_stats.npz',mean=mean,std=std)
    if args.model=='swin':
        torch.save({'state_dict':best_state,'classes':int(z['classes']),'mean':torch.from_numpy(mean),
                    'std':torch.from_numpy(std),'best_epoch':best_epoch},output/'checkpoint.pt')
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['sample_index','subject','context','trial','start','true','predicted'])
        for idx,p in zip(test,pred):
            w.writerow([idx,z['subject'][idx],z['context'][idx],z['trial'][idx],z['start'][idx],y[idx],p])
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--band',type=int,choices=range(5),default=4);p.add_argument('--classes',type=int,choices=[3,7],default=3)
    p=subs.add_parser('run');p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['swin','logreg','majority'],default='swin')
    p.add_argument('--protocol',choices=['within','loso'],default='loso')
    p.add_argument('--subject',type=int,default=15);p.add_argument('--context',choices=['vid','ima','multi'],default='multi')
    p.add_argument('--fold',type=int,choices=range(3),default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.data_root,args.output,args.band,args.classes)
    else:
        if args.epochs<1:parser.error('epochs must be positive')
        run(args)


if __name__=='__main__':main()
