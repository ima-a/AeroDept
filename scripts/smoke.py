"""Fast deterministic synthetic end-to-end software check, not research evidence."""
from pathlib import Path
import argparse,subprocess,sys

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts/synthetic_smoke'));a=p.parse_args();out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new output directory; do not overwrite prior outputs')
 out.mkdir(parents=True,exist_ok=True)
 def cli(*args):subprocess.run([sys.executable,'-m','aerodepth_lite.cli',*map(str,args)],check=True)
 data=out/'fixture';manifest=out/'manifest.json';config='configs/smoke.yaml'
 cli('make-fixture','--output',data)
 cli('inspect','--config',config,'--dataset-root',data,'--output',out/'inspection')
 cli('split','--dataset-root',data,'--output',manifest)
 cli('train','--config',config,'--dataset-root',data,'--manifest',manifest,'--output',out/'training')
 cli('replay','--config',config,'--dataset-root',data,'--manifest',manifest,'--checkpoint',out/'training/best.pt','--output',out/'replay')
 cli('report','--input',out/'replay','--output',out/'report')
 cli('infer','--image',data/'environment_0/trajectory_0000/rgb/frame_000000.JPEG','--output',out/'published_model_prediction.npy')
 (out/'VERIFIED_SYNTHETIC.txt').write_text('All stages ran successfully. Tiny-model optimization and all images here are synthetic software checks, not MidAir results. Published MobileNet weights were loaded only for inference.\n')
 print('Synthetic smoke passed:',out)
if __name__=='__main__':main()
