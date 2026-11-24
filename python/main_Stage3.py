"""Stage III entry point: few-shot personalization."""

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.plot_config import setup_matplotlib_backend
setup_matplotlib_backend('none')

from src.data_loader import determine_base_modalities, get_subject_ids
from src.train_Stage3 import run_finetune
from src.utils.console import clear_console
from src.utils.paths import output_root, resolve_h5_path, resolve_path


def load_params(config_path):
    with open(os.path.join(os.path.dirname(config_path), 'params.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    params = load_params(config_path)
    h5_path = resolve_h5_path(config)
    config['data']['h5_path'] = h5_path
    out_root = output_root(config)
    os.makedirs(out_root, exist_ok=True)

    exp_cfg = config['experiment']
    trunk = next(p for p in params['trunk_params'] if p['dts_type'] == exp_cfg['dts_type'])
    fusion = next(p for p in params['fusion_params'] if p['fusion_type'] == exp_cfg['fusion_type'])
    exp_cfg['trunk_params'] = copy.deepcopy(trunk)
    exp_cfg['fusion_params'] = copy.deepcopy(fusion)

    mods = determine_base_modalities(exp_cfg['modalities'])
    if exp_cfg['dts_type'] == 'dts' and (len(mods) < 2 or 'EEG' not in mods):
        exp_cfg['dts_type'] = 'no_dts'
        exp_cfg['trunk_params']['params']['NIRS']['use_dts'] = False

    seeds = exp_cfg.get('seeds', [42])
    lora_cfg = exp_cfg.get('lora_configs', [{'rank': 2, 'alpha': 8}])[0]
    exp_cfg['lora_rank'] = lora_cfg['rank']
    exp_cfg['lora_alpha'] = lora_cfg['alpha']
    target_modules = exp_cfg.get('target_modules_list', ['input_proj,linear1,linear2'])[0].split(',')

    grand = []
    for task in exp_cfg['task']:
        exp_cfg['task'] = task
        subject_ids = get_subject_ids(h5_path, task)
        if exp_cfg.get('subjects') not in (None, 'all'):
            subject_ids = [int(s) for s in exp_cfg['subjects']]
        pretrained_dir = resolve_path(exp_cfg['pretrained'][task])

        for method in exp_cfg['ft_method']:
            shots_list = [0] if method == 'none' else exp_cfg['shots']
            for shots in shots_list:
                exp_name = f'{task}_{method}_{shots}shot'
                exp_dir = os.path.join(out_root, exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                details_csv = os.path.join(exp_dir, 'details.csv')
                rows = []

                print(f'\n>>> {exp_name}')
                for target in subject_ids:
                    print(f'\n  --- {exp_name} | target {target} ---')
                    ckpt = os.path.join(pretrained_dir, f'target_{target}', 'best_model.pt')
                    if not os.path.exists(ckpt):
                        print(f'  missing checkpoint for target {target}, skip')
                        continue
                    seed_metrics = []
                    for seed in seeds:
                        metrics = run_finetune(config, target, ckpt, method, shots,
                                               seed, target_modules, output_dir=exp_dir)
                        if metrics:
                            seed_metrics.append(metrics)
                    if seed_metrics:
                        avg = pd.DataFrame(seed_metrics).mean().to_dict()
                        avg['target_subject'] = target
                        rows.append(avg)
                        print(f'  target {target}: acc={avg["accuracy"]:.4f}')
                        clear_console(config.get('logging', {}).get('clear_output', False))

                if not rows:
                    continue
                df = pd.DataFrame(rows)
                df.to_csv(details_csv, index=False)
                numeric = df.select_dtypes(include=[np.number])
                item = {'task': task, 'method': method, 'shots': shots,
                        'rank': exp_cfg['lora_rank'] if method in ('lora', 'dora') else 'N/A',
                        'alpha': exp_cfg['lora_alpha'] if method in ('lora', 'dora') else 'N/A'}
                for col in numeric.columns:
                    item[f'{col}_mean'] = numeric[col].mean()
                    item[f'{col}_std'] = numeric[col].std()
                grand.append(item)

    if grand:
        path = os.path.join(out_root, 'grand_summary.csv')
        pd.DataFrame(grand).to_csv(path, index=False)
        print(f'Grand summary: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Stage3_public.yaml')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
