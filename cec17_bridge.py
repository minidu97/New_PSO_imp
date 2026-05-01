
import ctypes
import os
import numpy as np
from typing import Callable

#LOAD THE SHARED LIBRARY

def _load_library() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))

    for name in ("cec17.so", "cec17.dll", "libcec17.so"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            _configure_signatures(lib)
            return lib

    raise FileNotFoundError(
        "Could not find cec17.so / cec17.dll next to cec17_bridge.py.\n"
        "Please run build_cec17.sh first (see instructions at top of this file)."
    )


def _configure_signatures(lib: ctypes.CDLL) -> None:
    #double cec17_call
    lib.cec17_call.restype  = ctypes.c_double
    lib.cec17_call.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # x array
        ctypes.c_int,                     # nx (dimension)
        ctypes.c_int,                     # func_num (1-29)
    ]


#Load once at import time
try:
    _lib = _load_library()
    _lib_loaded = True
except FileNotFoundError as _e:
    _lib = None
    _lib_loaded = False
    _load_error = str(_e)

#PUBLIC API

def cec17_func(func_id: int, dim: int) -> Callable[[np.ndarray], float]:
    if not _lib_loaded:
        raise RuntimeError(
            f"CEC17 library not loaded.\n{_load_error}"
        )

    valid_dims = {2, 10, 20, 30, 50, 100}
    if dim not in valid_dims:
        raise ValueError(f"dim must be one of {valid_dims}, got {dim}")
    if func_id < 1 or func_id > 29:
        raise ValueError(f"func_id must be 1-29, got {func_id}")
    if func_id == 2:
        raise ValueError("F2 has been deleted from CEC2017. Use func_id 1, 3-29.")

    def _evaluate(x: np.ndarray) -> float:
        # Ensure C-contiguous double array
        x_c = np.ascontiguousarray(x, dtype=np.float64)
        ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return float(_lib.cec17_call(ptr, ctypes.c_int(dim), ctypes.c_int(func_id)))

    _evaluate.__name__ = f"CEC2017_F{func_id}_D{dim}"
    return _evaluate


def cec17_error(fitness: float, func_id: int) -> float:
    return fitness - func_id * 100.0


def list_functions() -> None:
    names = {
        1:  "Shifted and Rotated Bent Cigar",
        2:  "DELETED",
        3:  "Shifted and Rotated Zakharov",
        4:  "Shifted and Rotated Rosenbrock",
        5:  "Shifted and Rotated Rastrigin",
        6:  "Shifted and Rotated Expanded Scaffer's F6",
        7:  "Shifted and Rotated Lunacek Bi-Rastrigin",
        8:  "Shifted and Rotated Non-Cont. Rastrigin",
        9:  "Shifted and Rotated Levy",
        10: "Shifted and Rotated Schwefel",
        11: "Hybrid Function 1  (Zakharov, Rosenbrock, Rastrigin)",
        12: "Hybrid Function 2  (Ellipsoid, Modified Schwefel, Bent Cigar)",
        13: "Hybrid Function 3  (Bent Cigar, Rosenbrock, LCBF)",
        14: "Hybrid Function 4  (HGBat, Diskus, Rosenbrock, Rastrigin)",
        15: "Hybrid Function 5  (Bent Cigar, HGBat, Rastrigin, Rosenbrock)",
        16: "Hybrid Function 6  (Zakharov, Rosenbrock, Schwefel, Ellipsoid)",
        17: "Hybrid Function 7",
        18: "Hybrid Function 8",
        19: "Hybrid Function 9",
        20: "Hybrid Function 10",
        21: "Composition Function 1  (Rosenbrock, HGBat)",
        22: "Composition Function 2  (HGBat, Rastrigin, Schwefel)",
        23: "Composition Function 3  (Schwefel, Rosenbrock, Ellipsoid)",
        24: "Composition Function 4  (Schwefel, HGBat, Rosenbrock, Discus)",
        25: "Composition Function 5  (HappyCat, HGBat, Rosenbrock, Schwefel)",
        26: "Composition Function 6  (Escaffer6, Schwefel, Griewank, Rosenbrock, Rastrigin)",
        27: "Composition Function 7  (HGBat, Rastrigin, Schwefel, Bent Cigar, Ellipsoid, Escaffer6)",
        28: "Composition Function 8  (Ackley, Griewank, Discus, Rosenbrock, HappyCat, Escaffer6)",
        29: "Composition Function 9",
    }
    print(f"{'ID':<4}  {'Function Name'}")
    print("-" * 60)
    for fid, name in names.items():
        skip = "  ← SKIP" if fid == 2 else ""
        print(f"F{fid:<3}  {name}{skip}")


#quick self test

if __name__ == "__main__":
    print("=" * 55)
    print("  CEC17 Bridge — Self Test")
    print("=" * 55)

    if not _lib_loaded:
        print(f"[FAIL] Library not loaded: {_load_error}")
        print("\nSteps to fix:")
        print("  1. Make sure input_data/ folder is in the same directory")
        print("  2. Run:  bash build_cec17.sh")
        print("  3. Re-run this test")
        exit(1)

    print("[OK] Library loaded successfully\n")

    dim = 10
    test_funcs = [1, 3, 4, 5, 10]   # skip F2 (deleted)

    print(f"{'Func':<6} {'Fitness':>16} {'Error':>16}")
    print("-" * 42)

    for fid in test_funcs:
        f  = cec17_func(fid, dim)
        x  = np.zeros(dim)           # test at origin
        fit = f(x)
        err = cec17_error(fit, fid)
        print(f"F{fid:<5} {fit:>16.6e} {err:>16.6e}")

    print("\n[PASS] All test functions evaluated successfully.")
    print("\nAvailable functions:")
    list_functions()