"""Portable offline loading of the published inference-only state dictionary."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .model import build_model,model_stats

def file_sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_inference_model(weights,metadata,device='cpu'):
 info=json.loads(Path(metadata).read_text())
 if file_sha256(weights)!=info['export_sha256']:raise ValueError('Exported-weight SHA-256 mismatch')
 config=info['training_config'];options=config['model']|{'pretrained':False,'min_depth':config['data']['min_depth']}
 model=build_model(options)
 state=torch.load(weights,map_location='cpu',weights_only=True)
 if not isinstance(state,dict) or not all(torch.is_tensor(v) for v in state.values()):raise ValueError('Expected plain inference state_dict, including buffers')
 model.load_state_dict(state,strict=True)
 if model_stats(model)['parameters']!=info['parameters']:raise ValueError('Parameter count mismatch')
 if not all(torch.isfinite(t).all() for t in state.values()):raise ValueError('Nonfinite inference tensor')
 return model.to(device).eval(),info

def image_tensor(path,image_size):
 with Image.open(path) as image:rgb=np.array(image.convert('RGB'),dtype=np.float32)/255.
 x=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
 return F.interpolate(x,size=tuple(image_size),mode='bilinear',align_corners=False)

@torch.inference_mode()
def infer_image(weights,metadata,image,output,device='cpu'):
 model,info=load_inference_model(weights,metadata,device)
 prediction=model(image_tensor(image,info['training_config']['data']['image_size']).to(device))[0,0].cpu().numpy()
 if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if output.suffix!='.npy':raise ValueError('Use .npy to preserve FP32 metre predictions')
 np.save(output,prediction)
 return {'output':str(output),'shape':list(prediction.shape),'units':'metres, Euclidean camera range','min':float(prediction.min()),'max':float(prediction.max()),'device':str(device),'weights_sha256':info['export_sha256'],'note':'Research depth estimate; no upper output clamp. Inference needs no MidAir download.'}
