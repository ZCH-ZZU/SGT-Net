"""Console helpers for notebook and terminal runs."""

def clear_console(enabled=False):
    """Clear the current notebook cell output when running under IPython.

    In a plain Python process this is a no-op, so the same configuration can be
    used from both .py entry points and .ipynb notebooks.
    """
    if not enabled:
        return
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        pass
