"""Export the trusted epoch-4 training checkpoint; never modifies its source."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch,yaml
from aerodepth_lite.inference import file_sha256,image_tensor,load_inference_model
from aerodepth_lite.model import build_model,load_checkpoint,model_contract,model_stats
EXPECTED='c1d4279cdaf594d1c15bce1f4f1b6ae322c7ec3ef28349df1dd47d975818ebd8'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--validation-results',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('models'));p.add_argument('--verification-image',type=Path);a=p.parse_args();torch.set_num_threads(2)
 if file_sha256(a.checkpoint)!=EXPECTED:raise ValueError('Only the frozen checkpoint is authorized for this export')
 cfg=yaml.safe_load(a.config.read_text());model=build_model(cfg['model']|{'pretrained':False,'min_depth':cfg['data']['min_depth']}).eval();ck=load_checkpoint(a.checkpoint,model,map_location='cpu')
 assert ck['epoch']==3 and ck['extra']['config']==cfg and ck['extra']['manifest_sha256']==file_sha256(a.manifest)
 a.output.mkdir(parents=True,exist_ok=True);weights=a.output/'mobilenet_v3_small_epoch4.state_dict.pt';meta=a.output/'mobilenet_v3_small_epoch4.json'
 if weights.exists() or meta.exists():raise FileExistsError('Refusing to overwrite a published export')
 torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},weights)
 info={'format':'plain torch state_dict; torch.load(weights_only=True), strict=True; includes parameters and buffers','architecture':'mobilenet_v3_small_depth','parameters':model_stats(model)['parameters'],'state_tensors':len(model.state_dict()),'original_checkpoint_sha256':EXPECTED,'export_sha256':file_sha256(weights),'export_bytes':weights.stat().st_size,'epoch_human':4,'selection_metric':'lowest recorded AMP full-validation AbsRel, 0.30312900253193636','training_config':cfg,'training_config_sha256':file_sha256(a.config),'manifest_sha256':file_sha256(a.manifest),'contract':model_contract(cfg),'output_convention':'FP32 positive Euclidean camera range in metres, lower floor0.1m; no upper output clamp. Evaluation alone masks finite GT[0.1,1000]m; near GT<10m.','pretrained_provenance':{'initialization':'torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1, ImageNet-1K; encoder subsequently finetuned on MidAir','url':'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth','sha256':'047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f','inference_downloads':False},'validation':json.loads(a.validation_results.read_text()),'training_duration_seconds':59281.158,'training_device':'NVIDIA GeForce GTX1650; FP32+AMP training','limitations':['Overfit after epoch4','Poor foggy near-field and far-depth geometry','Local floor saturation and excluded invalid GT regions','Only two selected validation trajectories; no final-test result','Pilot custom CNN has different data/protocol and is not a fair model comparison'],'notices':'See models/README.md and NOTICE.md for upstream ImageNet/TorchVision and MidAir terms/attribution; no blanket license grant is introduced.'}
 meta.write_text(json.dumps(info,indent=2)+'\n');restored,_=load_inference_model(weights,meta)
 generator=torch.Generator().manual_seed(42);inputs=[torch.rand((1,3,192,320),generator=generator),torch.zeros(1,3,192,320),torch.ones(1,3,192,320)]
 if a.verification_image:inputs.append(image_tensor(a.verification_image,[192,320]))
 differences=[]
 with torch.inference_mode():
  for x in inputs:
   original=model(x);exported=restored(x);torch.testing.assert_close(exported,original,atol=1e-6,rtol=1e-6);differences.append(float((exported-original).abs().max()))
 info['export_equivalence']={'precision':'CPU FP32 eval; identical inputs','absolute_tolerance':1e-6,'relative_tolerance':1e-6,'max_absolute_differences':differences,'inputs':'seed42 random, black, white; plus first validation RGB if supplied','passed':True};meta.write_text(json.dumps(info,indent=2)+'\n')
 (a.output/'SHA256SUMS').write_text(f"{file_sha256(weights)}  {weights.name}\n{file_sha256(meta)}  {meta.name}\n")
 print(json.dumps({'weights':str(weights),'bytes':info['export_bytes'],'sha256':info['export_sha256'],'equivalence':info['export_equivalence']},indent=2))
if __name__=='__main__':main()
