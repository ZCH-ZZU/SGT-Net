"""Shared training/evaluation helpers."""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def train_one_epoch(model, data_loader, criterion, optimizer, device,
                    epoch_num=1, total_epochs=1, l1_lambda=0.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(data_loader, desc=f'Epoch {epoch_num}/{total_epochs} [Train]', leave=False)
    for batch in pbar:
        data = {k: v.to(device) for k, v in batch['data'].items()}
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        if l1_lambda > 0:
            loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(batch['label'].numpy())
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(data_loader), float(np.mean(np.array(all_labels) == np.array(all_preds)))


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            data = {k: v.to(device) for k, v in batch['data'].items()}
            labels = batch['label'].to(device)
            outputs = model(data)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), total_loss / len(data_loader)


def average_scores(scores_list):
    if not scores_list:
        return {}
    avg = {}
    for key in scores_list[0].keys():
        values = [s[key] for s in scores_list]
        avg[f'{key}_mean'] = float(np.mean(values))
        avg[f'{key}_std'] = float(np.std(values))
    return avg


def save_time_course_excel(results_dir, exp_name, all_time_courses):
    filepath = os.path.join(results_dir, f'{exp_name}_TestResults.xlsx')
    print(f'  -> Saving time-course details to {filepath}')
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        for sub_id, matrix in all_time_courses.items():
            if matrix is None:
                continue
            pd.DataFrame(matrix).to_excel(writer, sheet_name=f'sub_{sub_id}', index=False)
        valid = [m for m in all_time_courses.values() if m is not None]
        if valid:
            min_len = min(m.shape[0] for m in valid)
            if min_len > 0:
                curves = [np.mean(m[:min_len], axis=0) for m in valid]
                pd.DataFrame(np.vstack(curves)).to_excel(
                    writer, sheet_name='MeanCurve', index=False)
