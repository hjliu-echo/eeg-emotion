"""Meaningful checks for leakage prevention and the corrected upstream mask."""
from pathlib import Path
import sys
import tempfile
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline import split_indices,fit_stats,build_swin


def main():
    import torch
    torch.set_num_threads(2)
    rng=np.random.default_rng(42)
    # Multiple windows share a trial and must never cross a split boundary.
    subject=np.repeat(np.arange(1,5),2*21*2)
    context=np.tile(np.repeat(['ima','vid'],21*2),4)
    trial=np.tile(np.repeat(np.arange(21),2),8)
    x=rng.normal(size=(len(subject),4,32,32)).astype(np.float32)
    for classes in [3,7]:
        y=trial//3
        if classes==3:y=np.array([0,0,0,1,2,2,2])[y]
        z=dict(x=x,y=y,subject=subject,context=context,trial=trial,
               start=np.tile([0,4],len(subject)//2),classes=classes,band=4,synthetic=True)
        for ctx in ['ima','vid','multi']:
            for protocol in ['within','loso']:
                for fold in range(3) if protocol=='within' else [0]:
                    tr,va,te=split_indices(z,protocol,1,ctx,fold,42)
                    mu,sd=fit_stats(x,tr)
                    altered=x.copy();altered[np.r_[va,te]]+=1000
                    mu2,sd2=fit_stats(altered,tr)
                    assert np.array_equal(mu,mu2) and np.array_equal(sd,sd2)
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'synthetic.npz';np.savez_compressed(cache,**z)
            for model in ['majority','logreg']:
                p=subprocess.run([sys.executable,str(ROOT/'src/pipeline.py'),'run',
                     '--cache',str(cache),'--output',str(Path(directory)/model),
                     '--model',model,'--subject','1'],capture_output=True,text=True)
                assert p.returncode==0,p.stderr
                report=json.loads((Path(directory)/model/'metrics.json').read_text())
                assert report['data_status']=='synthetic_check_only'
                assert sum(map(sum,report['confusion_matrix']))==report['counts']['test']
    sys.path.insert(0,str(ROOT/'third_party/st_shap'))
    from SWIN_trans import drop_path_f
    mask=drop_path_f(torch.ones(128,2,2),drop_prob=.5,training=True)
    assert set(mask.unique().tolist())=={0.,2.}
    net=build_swin(3)
    net.train();result=net(torch.randn(2,4,32,32))
    assert result.shape==(2,3) and torch.isfinite(result).all()
    torch.nn.functional.cross_entropy(result,torch.tensor([0,2])).backward()
    print('PASS: 24 grouped split checks; held-out perturbation; baseline CLI (3/7 classes); stochastic-depth mask; Swin forward/backward. Synthetic only.')


if __name__=='__main__':main()
