"""Stage II entry point: LOSO adaptation with optional source selection."""

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.plot_config import setup_matplotlib_backend
setup_matplotlib_backend('none')

from src.data_loader import determine_base_modalities, get_subject_ids
from src.train_Stage2 import run_loso
from src.utils.console import clear_console
from src.utils.paths import output_root, resolve_h5_path
from src.utils.selection_utils import get_selection_plan
from src.utils.training_utils import save_time_course_excel


def load_params(config_path):
    with open(os.path.join(os.path.dirname(config_path), 'params.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def selection_plans(sel_cfg):
    if not sel_cfg.get('enabled', False):
        return [('none', 'hard', 'all')]
    plans = []
    if sel_cfg.get('run_baseline_control', True):
        plans.append(('baseline', 'hard', 'all'))
    for method in sel_cfg.get('methods', []):
        for mode in sel_cfg.get('modes', []):
            for k in sel_cfg.get('top_k', {}).get(mode, ['all']):
                plans.append((method, mode, k))
    return plans


def append_csv(df, path):
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


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

    seed = config.get('seed', 42)
    grand = []

    for task in exp_cfg['task']:
        exp_cfg['task'] = task
        subject_ids = get_subject_ids(h5_path, task)
        if exp_cfg.get('subjects') not in (None, 'all'):
            subject_ids = [int(s) for s in exp_cfg['subjects']]
        if not subject_ids:
            continue

        for strategy in exp_cfg['strategies']:
            exp_cfg['strategy_current'] = strategy
            for method, mode, k in selection_plans(exp_cfg['source_selection']):
                if method in ('none', 'baseline'):
                    exp_name = f'{task}_{strategy}'
                else:
                    exp_name = f'{task}_{strategy}_{method}_{mode}_K{k}'
                exp_dir = os.path.join(out_root, exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                summary_csv = os.path.join(exp_dir, 'LOSO_Summary.csv')
                print(f'\n>>> {exp_name}')

                rows, tcs = [], {}
                for target in subject_ids:
                    print(f'\n  --- {exp_name} | target {target} ---')
                    candidates = [s for s in subject_ids if s != target]
                    weights = None
                    if method not in ('none', 'baseline'):
                        sources, weights, _ = get_selection_plan(
                            config, target, candidates, method, mode, k,
                            device='cuda' if torch.cuda.is_available() else 'cpu')
                    else:
                        sources = candidates
                    if not sources:
                        continue
                    metrics, tc = run_loso(config, target, sources, weights,
                                           output_dir=exp_dir, seed=seed)
                    if metrics is None:
                        continue
                    metrics['target_subject'] = target
                    rows.append(metrics)
                    if tc is not None:
                        tcs[target] = tc
                    append_csv(pd.DataFrame([metrics]), summary_csv)
                    print(f'  target {target}: acc={metrics["accuracy"]:.4f}')
                    clear_console(config.get('logging', {}).get('clear_output', False))

                if not rows:
                    continue
                df = pd.DataFrame(rows)
                numeric = df.select_dtypes(include=[np.number])
                mean_row = numeric.mean().to_dict()
                std_row = numeric.std().to_dict()
                mean_row['target_subject'] = 'Mean'
                std_row['target_subject'] = 'Std'
                pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True).to_csv(
                    summary_csv, index=False)
                if tcs:
                    save_time_course_excel(exp_dir, exp_name, tcs)

                item = {'task': task, 'strategy': strategy, 'method': method,
                        'mode': mode, 'k': k, 'subjects': len(df)}
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
    parser.add_argument('--config', default='config/Stage2_public.yaml')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
