"""Explicit validation stages. Existing completed outputs are never overwritten."""
from pathlib import Path
import argparse,json
from aerodepth_lite.presentation import main


def guarded_main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('stage');p.add_argument('--config',type=Path,default=Path('configs/presentation_local.json'))
    args,_=p.parse_known_args()
    if args.config.is_file():
        config=json.loads(args.config.read_text());result=Path(config['results'])
        root=Path(config['dataset_root']).resolve()
        for field in ['results','cache','export_directory']:
            target_path=Path(config[field]).resolve()
            if target_path==root or root in target_path.parents:
                raise SystemExit('Presentation output cannot be inside the dataset: '+field)
        target={'quality':'scheduler_results.json','cache-check':'live_cache_equivalence.json','timing':'timing_summary.json'}.get(args.stage)
        if target and (result/target).exists():
            raise SystemExit('Completed evidence exists; use a fresh frozen output/config for reproduction: '+str(result/target))
    main()

if __name__=='__main__':guarded_main()
