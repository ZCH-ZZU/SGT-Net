"""Path helpers shared by the Stage entry points."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / 'python'


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    # Prefer paths relative to python/ so that 'results/...' always refers to
    # python/results, matching output_root(). Data paths fall back to the repo
    # root (e.g. datasets/, data/physio_metadata.csv).
    for base in (PYTHON_ROOT, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return str(candidate)
    return str(PYTHON_ROOT / path)


def resolve_h5_path(config):
    return resolve_path(config['data']['h5_path'])


def output_root(config, default='results'):
    path = Path(config.get('output_dir', default))
    if path.is_absolute():
        return str(path)
    return str(PYTHON_ROOT / path)
