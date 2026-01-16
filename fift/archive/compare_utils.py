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
    
def check_precomputed_gl_files(n_gl, Umax, directory):
    if not directory:
        raise RuntimeError("FIFT_GL2D_DIR is not set.")
    path_base = f"gl2d_n{n_gl}_U{int(Umax)}"
    required = [
        f"{path_base}.u1.npy",
        f"{path_base}.u2.npy",
        f"{path_base}.W.npy",
        f"{path_base}.meta.json",
    ]
    missing = []
    for fname in required:
        if not pathlib.Path(directory, fname).exists():
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"The following precomputed GL2D files are missing in {directory}:\n  "
            + "\n  ".join(missing)
        )
    else:
        print(f"All precomputed GL2D files found in {directory} for n={n_gl}, Umax={Umax}.\n")
