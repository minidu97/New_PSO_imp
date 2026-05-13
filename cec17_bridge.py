import ctypes
import os
import numpy as np
from typing import Callable


#LOAD THE SHARED LIBRARY

def _load_library() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))

    # macOS uses .dylib, Linux uses .so
    for name in ("cec17.so", "cec17.dylib", "libcec17.so"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            _configure_signatures(lib)
            return lib

    raise FileNotFoundError(
        "Could not find cec17.so next to cec17_bridge.py.\n"
        "Run:  bash build_cec17.sh"
    )


def _configure_signatures(lib: ctypes.CDLL) -> None:
    # double cec17_call(double *x, int nx, int func_num)
    lib.cec17_call.restype  = ctypes.c_double
    lib.cec17_call.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.c_int,                     # nx (dimension)
        ctypes.c_int,                     # func_num (1-29)
    ]


# Load once at import time
try:
    _lib = _load_library()
    _lib_loaded = True
    _load_error = ""
except FileNotFoundError as e:
    _lib = None
    _lib_loaded = False
    _load_error = str(e)


#PUBLIC API

def cec17_func(func_id: int, dim: int) -> Callable[[np.ndarray], float]:
    if not _lib_loaded:
        raise RuntimeError(f"CEC17 library not loaded.\n{_load_error}")

    valid_dims = {2, 10, 20, 30, 50, 100}
    if dim not in valid_dims:
        raise ValueError(f"dim must be one of {valid_dims}, got {dim}")
    if func_id < 1 or func_id > 29:
        raise ValueError(f"func_id must be 1-29, got {func_id}")

    def _evaluate(x: np.ndarray) -> float:
        x_c = np.ascontiguousarray(x, dtype=np.float64)
        ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return float(_lib.cec17_call(ptr, ctypes.c_int(dim), ctypes.c_int(func_id)))

    _evaluate.__name__ = f"CEC2017_F{func_id}_D{dim}"
    return _evaluate


def cec17_error(fitness: float, func_id: int) -> float:
    return fitness - func_id * 100.0


# ─────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  CEC17 Bridge — Self Test")
    print("=" * 50)

    if not _lib_loaded:
        print(f"[FAIL] Library not loaded:\n{_load_error}")
        print("\nFix:")
        print("  1. Make sure cec17_test_func.c, cec17.c, cec17.h are here")
        print("  2. Make sure input_data/ folder is here")
        print("  3. Run:  bash build_cec17.sh")
        exit(1)

    print("[OK] cec17.so loaded\n")

    dim = 10
    print(f"{'Func':<6} {'Fitness':>16} {'Error':>16}")
    print("-" * 40)

    for fid in [1, 2, 3, 4, 5, 10]:
        f   = cec17_func(fid, dim)
        x   = np.zeros(dim)
        fit = f(x)
        err = cec17_error(fit, fid)
        print(f"F{fid:<5} {fit:>16.6e} {err:>16.6e}")

    print("\n[PASS] All functions evaluated successfully.")
    print("You can now run:  python pso.py")