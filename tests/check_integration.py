from pathlib import Path
import sys,tempfile,argparse,json
import numpy as np
import torch
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'src'))
from pipeline import prepare,run
torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    # Exercise provider-schema preprocessing on synthetic DE, including labels.
    np.save(root/'sub-01_ses-ima_task-emotion_de.npy',np.ones((64,630,5),np.float32))
    prepare(root,root/'prepared.npz',4,3)
    with np.load(root/'prepared.npz') as z:
        assert z['x'].shape==(147,4,32,32)
        assert np.bincount(z['y']).tolist()==[63,21,63]
        assert np.unique(z['trial']).size==21
    # Execute full Swin checkpoint-selection and test-output path on synthetic data.
    rng=np.random.default_rng(42)
    subject=np.repeat(np.arange(1,4),21)
    trial=np.tile(np.arange(21),3)
    np.savez_compressed(root/'synthetic.npz',
        x=rng.normal(size=(63,4,32,32)).astype(np.float32),
        y=np.array([0,0,0,1,2,2,2])[trial//3],subject=subject,
        context=np.repeat('ima',63),trial=trial,start=np.zeros(63,int),
        classes=3,band=4,synthetic=True)
    args=argparse.Namespace(cache=str(root/'synthetic.npz'),output=str(root/'run'),
        seed=42,model='swin',protocol='loso',subject=1,context='ima',fold=0,epochs=1,batch_size=4)
    run(args)
    report=json.loads((root/'run/metrics.json').read_text())
    ckpt=torch.load(root/'run/checkpoint.pt',map_location='cpu',weights_only=True)
    assert report['data_status']=='synthetic_check_only'
    assert ckpt['best_epoch']==1
    assert (root/'run/predictions.csv').exists()
    print('PASS: synthetic DE preparation, 3:1:3 labels, full one-epoch Swin run, checkpoint reload and prediction export.')
