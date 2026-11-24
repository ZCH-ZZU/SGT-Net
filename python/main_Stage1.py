"""Stage I entry point: intra-subject 5-fold GroupKFold."""

import argparse
import copy
import itertools
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.plot_config import setup_matplotlib_backend
setup_matplotlib_backend('none')

from src.data_loader import determine_base_modalities, get_subject_ids
from src.train_Stage1 import run_single_subject
from src.utils.console import clear_console
from src.utils.paths import output_root, resolve_h5_path
from src.utils.training_utils import save_time_course_excel


def load_params(config_path):
    path = os.path.join(os.path.dirname(config_path), 'params.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_experiments(config, params):
    axes = config['experiment_axes']
    keys, values = zip(*axes.items())
    experiments = []
    for combo in itertools.product(*values):
        exp = dict(zip(keys, combo))
        trunk = next((p for p in params['trunk_params'] if p['dts_type'] == exp['dts_type']), None)
        fusion = next((p for p in params['fusion_params'] if p['fusion_type'] == exp['fusion_type']), None)
        if trunk is None or fusion is None:
            continue
        exp['trunk_params'] = copy.deepcopy(trunk)
        exp['fusion_params'] = copy.deepcopy(fusion)
        experiments.append(exp)
    return experiments


def channels_from_h5(h5_path, task, subject_ids, modalities):
    base = determine_base_modalities(modalities)
    with h5py.File(h5_path, 'r') as f:
        group = f[f'{task}/subject_{subject_ids[0]:02d}']
        return {mod: {'channels': group[f'{mod.lower()}_data'].shape[1]} for mod in base}


def run(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    params = load_params(config_path)
    h5_path = resolve_h5_path(config)
    out_dir = output_root(config)
    os.makedirs(out_dir, exist_ok=True)

    experiments = build_experiments(config, params)
    print(f'Stage I: {len(experiments)} experiment(s)')

    grand = []
    for i, exp in enumerate(experiments, 1):
        config['experiment'] = exp
        task = exp['task']
        subject_ids = get_subject_ids(h5_path, task)
        if exp['subjects'] != 'all':
            subject_ids = [int(s) for s in exp['subjects']]
        if not subject_ids:
            print(f'  No subjects for task {task}.')
            continue

        mods = determine_base_modalities(exp['modalities'])
        # DTS needs an EEG anchor and at least two modalities.
        if exp['dts_type'] == 'dts' and (len(mods) < 2 or 'EEG' not in mods):
            exp['trunk_params']['params']['NIRS']['use_dts'] = False
        channels = channels_from_h5(h5_path, task, subject_ids, exp['modalities'])

        mod_str = '+'.join(exp['modalities']) if isinstance(exp['modalities'], list) else exp['modalities']
        exp_name = f'{task}_{exp["dts_type"]}_{mod_str}_{exp["fusion_type"]}'
        exp_dir = os.path.join(out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        summary_csv = os.path.join(exp_dir, f'{exp_name}_Summary.csv')

        print(f'\n[{i}/{len(experiments)}] {exp_name}')
        rows, tcs = [], {}
        for sid in subject_ids:
            result, _, tc = run_single_subject(config, int(sid), channels, h5_path, out_dir)
            if result:
                rows.append(result)
                if tc is not None:
                    tcs[int(sid)] = tc
            clear_console(config.get('logging', {}).get('clear_output', False))

        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(summary_csv, index=False)
        if tcs:
            save_time_course_excel(exp_dir, exp_name, tcs)

        numeric = df.select_dtypes(include=[np.number])
        mean_row = numeric.mean().to_dict()
        std_row = numeric.std().to_dict()
        mean_row['subject_id'] = 'Mean'
        std_row['subject_id'] = 'Std'
        pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True).to_csv(
            summary_csv, index=False)

        item = {'task': task, 'dts_type': exp['dts_type'],
                'fusion_type': exp['fusion_type'], 'modalities': mod_str,
                'subjects': len(df)}
        for col in numeric.columns:
            if col.endswith('_mean'):
                item[col.replace('_mean', '_grand_mean')] = numeric[col].mean()
                item[col.replace('_mean', '_grand_std')] = numeric[col].std()
        grand.append(item)

    if grand:
        path = os.path.join(out_dir, 'grand_summary.csv')
        pd.DataFrame(grand).to_csv(path, index=False)
        print(f'Grand summary: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Stage1_public.yaml')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
