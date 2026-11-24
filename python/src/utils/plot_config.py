"""Matplotlib backend selection."""


def setup_matplotlib_backend(plot_mode='inline'):
    """Configure matplotlib for inline, window, or headless use.

    ``plot_mode='none'`` forces the Agg backend, which is the safe choice for
    servers and notebooks that only save figures to disk.
    """
    import matplotlib

    if plot_mode == 'none':
        matplotlib.use('Agg', force=True)
        return

    try:
        if plot_mode == 'inline':
            try:
                from IPython import get_ipython
                get_ipython().run_line_magic('matplotlib', 'inline')
                print('Plot mode: inline')
            except Exception:
                matplotlib.use('Agg')
                print('Plot mode: inline (using Agg backend)')

        elif plot_mode == 'window':
            for backend in ['TkAgg', 'Qt5Agg', 'QtAgg']:
                try:
                    matplotlib.use(backend)
                    print(f'Plot mode: window ({backend})')
                    break
                except Exception:
                    continue
            else:
                matplotlib.use('Agg')
                print('Plot mode: window (fallback to Agg)')

    except Exception as exc:
        print(f'Plot setup error: {exc}, using default backend')
