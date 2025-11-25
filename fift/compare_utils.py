import time
import numpy as np

def time_func(fn, *args, repeat=3, warmup=1, **kwargs):
    """
    Time a callable with given args/kwargs. Returns best-of-N (s).
    """
    for _ in range(max(0, warmup)):
        fn(*args, **kwargs)
    best = float("inf")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best

def align_global_phase(a, b):
    """
    Return a * exp(-i*phi) so that mean phase(a) ~ mean phase(b).
    """
    m = np.mean(a / np.maximum(1e-300, b))
    phi = np.angle(m)
    return a * np.exp(-1j*phi), phi

def plot_overlays_ws(ws, F_fift, F_glow, N_gl, Umax, title="F(w) overlay", align_phase=True):
    """
    Quick overlay plot for scalar F(w) from FIFT vs GLoW.
    """
    import matplotlib.pyplot as plt
    if align_phase:
        F_fift_al, _ = align_global_phase(F_fift, F_glow)
    else:
        F_fift_al = F_fift
    plt.figure()
    plt.loglog(ws, np.abs(F_fift_al), '-', label='FIFT')
    plt.loglog(ws, np.abs(F_glow), '--', label='GLoW')
    plt.xlabel('w'); plt.ylabel('|F(w)|'); plt.legend();
    plt.title(title + f'; N={N_gl}, U={Umax}')
    plt.show()