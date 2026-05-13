import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

from cec17_bridge import cec17_func, cec17_error

# All results are saved into this folder automatically
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


#CONFIGURATION

@dataclass
class PSOConfig:

    NP: int   = 30          # Number of particles
    dim: int  = 10          # Problem dimensionality
    T: int    = 500         # Max iterations

    # Search space
    lb: float = -100.0      # Lower bound
    ub: float =  100.0      # Upper bound

    # Standard PSO coefficients
    w:  float = 0.9         # Inertia weight (decreases linearly to w_min)
    w_min: float = 0.4      # Minimum inertia weight
    c1: float = 2.0         # Cognitive coefficient
    c2: float = 2.0         # Social coefficient

    # Archive (Top-M)
    p: float  = 0.1         # Archive fraction  →  M = floor(p * NP)
    p_min: float = 0.05     # Minimum archive fraction
    diversity_threshold: float = 1e6  # Fitness std below this and then it shrink archive

    # phi parameter
    phi: float = 0.5        # Blending parameter φ ∈ [0, 1]
    mutation_scale: float = 0.01  # Gaussian noise streangth 

    # CEC 2017
    cec_func_id: int = 1    # CEC 2017 function ID (1–29)
    cec_runs: int    = 30   # Independent runs for statistics

    @property
    def M(self) -> int:
        return max(1, int(np.floor(self.p * self.NP)))


#PARTICLE

@dataclass
class Particle:
    position:  np.ndarray
    velocity:  np.ndarray
    pbest_pos: np.ndarray
    pbest_fit: float = np.inf   # Minimisation

    def update_pbest(self, fitness: float) -> None:
        if fitness < self.pbest_fit:
            self.pbest_fit = fitness
            self.pbest_pos = self.position.copy()


#INITIALISATION

def initialise_swarm(cfg: PSOConfig) -> List[Particle]:
    particles = []
    span = cfg.ub - cfg.lb

    for _ in range(cfg.NP):
        pos = np.random.uniform(cfg.lb, cfg.ub, cfg.dim)
        vel = np.random.uniform(-span, span, cfg.dim)
        p   = Particle(
            position  = pos.copy(),
            velocity  = vel.copy(),
            pbest_pos = pos.copy(),
            pbest_fit = np.inf
        )
        particles.append(p)

    return particles


def evaluate_swarm(particles: List[Particle],
                   func: Callable) -> np.ndarray:
    return np.array([func(p.position) for p in particles])


def get_gbest(particles: List[Particle]) -> Tuple[np.ndarray, float]:
    best_particle = min(particles, key=lambda p: p.pbest_fit)
    return best_particle.pbest_pos.copy(), best_particle.pbest_fit


#PHASE 1 — STANDARD PSO

def run_standard_pso(func: Callable,
                     cfg:  PSOConfig) -> Tuple[np.ndarray, float, List[float]]:
    particles = initialise_swarm(cfg)
    fitnesses = evaluate_swarm(particles, func)

    # Initialise pbest
    for p, f in zip(particles, fitnesses):
        p.pbest_fit = f
        p.pbest_pos = p.position.copy()

    gbest_pos, gbest_fit = get_gbest(particles)
    curve = []

    for t in range(cfg.T):

        # Linearly decrease inertia weight
        w = cfg.w - (cfg.w - cfg.w_min) * (t / cfg.T)

        for p in particles:
            r1 = np.random.rand(cfg.dim)
            r2 = np.random.rand(cfg.dim)

            # Velocity update: v_i = w*v_i + c1*r1*(pbest_i-x_i) + c2*r2*(gbest-x_i)
            p.velocity = (  w * p.velocity
                          + cfg.c1 * r1 * (p.pbest_pos - p.position)
                          + cfg.c2 * r2 * (gbest_pos   - p.position) )

            # Clamp velocity to prevent particles flying out of bounds
            v_max = (cfg.ub - cfg.lb) * 0.2
            p.velocity = np.clip(p.velocity, -v_max, v_max)

            # Position update: x_i = x_i + v_i
            p.position = p.position + p.velocity
            p.position = np.clip(p.position, cfg.lb, cfg.ub)

            # Evaluate & update pbest
            fit = func(p.position)
            p.update_pbest(fit)

        # Update gbest
        gbest_pos, gbest_fit = get_gbest(particles)
        curve.append(gbest_fit)

    return gbest_pos, gbest_fit, curve


#PHASE 2 — ELITE ARCHIVE (TOP-M)

def build_archive(particles: List[Particle],
                  M: int,
                  cfg: PSOConfig) -> Tuple[List[Particle], float]:
    diversity = float(np.std([p.pbest_fit for p in particles]))

    if diversity < cfg.diversity_threshold:
        M_adaptive = max(1, int(np.floor(cfg.p_min * cfg.NP)))
    else:
        M_adaptive = M

    sorted_particles = sorted(particles, key=lambda p: p.pbest_fit)
    archive   = sorted_particles[:M_adaptive]
    f_worst_M = archive[-1].pbest_fit
    return archive, f_worst_M


def run_modified_pso(func: Callable,
                     cfg:  PSOConfig) -> Tuple[np.ndarray, float, List[float]]:
    particles = initialise_swarm(cfg)
    fitnesses = evaluate_swarm(particles, func)

    for p, f in zip(particles, fitnesses):
        p.pbest_fit = f
        p.pbest_pos = p.position.copy()

    gbest_pos, gbest_fit = get_gbest(particles)
    curve = []

    for t in range(cfg.T):

        w = cfg.w - (cfg.w - cfg.w_min) * (t / cfg.T)

        # Build elite archive this iteration
        archive, f_worst_M = build_archive(particles, cfg.M, cfg)

        for p in particles:

            fit_current = func(p.position)

            if fit_current <= f_worst_M:
                continue

            exp2 = archive[np.random.randint(len(archive))].pbest_pos

            r1 = np.random.rand(cfg.dim)
            r2 = np.random.rand(cfg.dim)

            # Velocity update with φ scaling the social term
            p.velocity = (  w * p.velocity
                          + cfg.c1 * r1           * (p.pbest_pos - p.position)
                          + cfg.phi * cfg.c2 * r2 * (exp2        - p.position) )

            # Clamp velocity
            v_max = (cfg.ub - cfg.lb) * 0.2
            p.velocity = np.clip(p.velocity, -v_max, v_max)

            # Position update
            p.position = p.position + p.velocity
            p.position = np.clip(p.position, cfg.lb, cfg.ub)
            sigma = (cfg.ub - cfg.lb) * cfg.mutation_scale * (1 - t / cfg.T)
            p.position += np.random.normal(0, sigma, cfg.dim)
            p.position  = np.clip(p.position, cfg.lb, cfg.ub)

            fit_new = func(p.position)
            p.update_pbest(fit_new)

        gbest_pos, gbest_fit = get_gbest(particles)
        curve.append(gbest_fit)

    return gbest_pos, gbest_fit, curve


#PHASE 3 — φ PARAMETER TUNING

def tune_phi(func: Callable,
             cfg:  PSOConfig,
             phi_values: Optional[List[float]] = None) -> dict:

    if phi_values is None:
        phi_values = [round(v, 1) for v in np.arange(0.1, 1.0, 0.1)]

    results = {}

    for phi in phi_values:
        cfg.phi = phi
        run_results = []

        for _ in range(cfg.cec_runs):
            _, best_fit, _ = run_modified_pso(func, cfg)
            run_results.append(best_fit)

        results[phi] = run_results
        print(f"  φ = {phi:.1f}  |  mean = {np.mean(run_results):.4e}"
              f"  |  std = {np.std(run_results):.4e}")

    return results


#PHASE 4 — CEC 2017 BENCHMARK EVALUATION

# All 29 CEC2017 functions — your new cec17_test_func.c supports all of them
CEC17_FUNC_IDS = list(range(1, 30))  # [1, 2, 3, ... 29]


def load_cec2017_function(func_id: int, dim: int) -> Callable:
    return cec17_func(func_id, dim)


def run_cec2017_benchmark(cfg: PSOConfig) -> dict:

    report = {}

    for func_id in CEC17_FUNC_IDS:
        cfg.cec_func_id = func_id
        func = load_cec2017_function(func_id, cfg.dim)

        std_results = []
        mod_results = []

        print(f"\n── CEC2017 F{func_id:02d} ──")

        for run in range(cfg.cec_runs):
            _, std_fit, _ = run_standard_pso(func, cfg)
            _, mod_fit, _ = run_modified_pso(func, cfg)
            std_results.append(std_fit)
            mod_results.append(mod_fit)
            print(f"  run {run+1:02d}/{cfg.cec_runs}  "
                  f"std={std_fit:.4e}  mod={mod_fit:.4e}")

        report[func_id] = {
            'standard': {
                'mean':     np.mean(std_results),
                'std':      np.std(std_results),
                'all_runs': std_results
            },
            'modified': {
                'mean':     np.mean(mod_results),
                'std':      np.std(mod_results),
                'all_runs': mod_results
            }
        }

        print(f"  Standard PSO  |  mean={np.mean(std_results):.4e}  "
              f"std={np.std(std_results):.4e}")
        print(f"  Modified PSO  |  mean={np.mean(mod_results):.4e}  "
              f"std={np.std(mod_results):.4e}")

    return report


#PLOTTING UTILITIES

def plot_convergence(curves: dict, title: str = "Convergence Curve",
                     filename: str = "convergence.png") -> None:
    plt.figure(figsize=(9, 5))
    for label, curve in curves.items():
        plt.plot(curve, label=label)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    print(f"  [Saved] {path}")
    plt.show()


def plot_phi_comparison(results: dict) -> None:
    labels = [f"φ={k}" for k in results.keys()]
    data   = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.yscale("log")
    plt.xlabel("φ value")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Modified PSO: φ Parameter Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "phi_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"  [Saved] {path}")
    plt.show()

    # Save phi results to CSV
    csv_path = os.path.join(RESULTS_DIR, "phi_tuning_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phi", "mean", "std"] + [f"run_{i+1}" for i in range(len(data[0]))])
        for phi, runs in results.items():
            writer.writerow([phi, round(float(np.mean(runs)), 8),
                             round(float(np.std(runs)), 8)] + [round(float(r), 8) for r in runs])
    print(f"  [Saved] {csv_path}")


def plot_benchmark_comparison(report: dict) -> None:
    func_ids  = list(report.keys())
    std_means = [report[f]['standard']['mean'] for f in func_ids]
    mod_means = [report[f]['modified']['mean']  for f in func_ids]

    x     = np.arange(len(func_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width/2, std_means, width, label='Standard PSO', color='steelblue')
    ax.bar(x + width/2, mod_means, width, label='Modified PSO', color='tomato')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in func_ids], rotation=45)
    ax.set_ylabel("Mean Best Fitness (log scale)")
    ax.set_title("CEC 2017: Standard PSO vs Modified PSO")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cec17_benchmark_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"  [Saved] {path}")
    plt.show()
    save_benchmark_results(report)


def save_benchmark_results(report: dict) -> None:

    #CSV: one row per function
    csv_path = os.path.join(RESULTS_DIR, "cec17_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "func_id",
            "std_mean", "std_std",
            "mod_mean", "mod_std",
            "better"        # which algorithm had lower mean fitness
        ])
        for fid, data in report.items():
            std_mean = data['standard']['mean']
            mod_mean = data['modified']['mean']
            better   = "Modified" if mod_mean < std_mean else "Standard"
            writer.writerow([
                f"F{fid}",
                f"{std_mean:.6e}", f"{data['standard']['std']:.6e}",
                f"{mod_mean:.6e}", f"{data['modified']['std']:.6e}",
                better
            ])
    print(f"  [Saved] {csv_path}")

    #JSON: full raw data including all runs
    json_path = os.path.join(RESULTS_DIR, "cec17_benchmark_results.json")
    serialisable = {
        str(fid): {
            "standard": {
                "mean":     round(data["standard"]["mean"], 8),
                "std":      round(data["standard"]["std"],  8),
                "all_runs": [round(v, 8) for v in data["standard"]["all_runs"]]
            },
            "modified": {
                "mean":     round(data["modified"]["mean"], 8),
                "std":      round(data["modified"]["std"],  8),
                "all_runs": [round(v, 8) for v in data["modified"]["all_runs"]]
            }
        }
        for fid, data in report.items()
    }
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  [Saved] {json_path}")

    #Print summary table to terminal
    print()
    print(f"  {'Func':<6} {'Std Mean':>14} {'Std Std':>14} "
          f"{'Mod Mean':>14} {'Mod Std':>14}  {'Better'}")
    print("  " + "-" * 72)
    for fid, data in report.items():
        std_mean = data['standard']['mean']
        mod_mean = data['modified']['mean']
        better   = "Modified ✓" if mod_mean < std_mean else "Standard ✓"
        print(f"  F{fid:<5} {std_mean:>14.4e} {data['standard']['std']:>14.4e} "
              f"{mod_mean:>14.4e} {data['modified']['std']:>14.4e}  {better}")


#SIMPLE TEST FUNCTIONS (for Phases 1-3, no CEC17 needed)

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


#MAIN

def main():
    cfg = PSOConfig(NP=30, dim=10, T=500)

    #PHASE 1 — Standard PSO on Sphere function
    print("=" * 60)
    print("  PHASE 1 — Standard PSO on Sphere function")
    print("=" * 60)

    gbest_pos, gbest_fit, curve_std = run_standard_pso(sphere, cfg)
    print(f"  Best fitness : {gbest_fit:.6e}")
    plot_convergence({"Standard PSO": curve_std},
                     "Phase 1: Standard PSO — Sphere",
                     filename="phase1_convergence_sphere.png")

    #PHASE 2 — Modified PSO (Elite Archive) on Sphere
    print("\n" + "=" * 60)
    print("  PHASE 2 — Modified PSO (Elite Archive) on Sphere")
    print("=" * 60)

    gbest_pos, gbest_fit, curve_mod = run_modified_pso(sphere, cfg)
    print(f"  Best fitness : {gbest_fit:.6e}")
    plot_convergence(
        {"Standard PSO": curve_std, "Modified PSO": curve_mod},
        "Phase 1 vs Phase 2: Sphere",
        filename="phase2_convergence_sphere.png"
    )

    #PHASE 3 — φ Parameter Tuning on Sphere
    print("\n" + "=" * 60)
    print("  PHASE 3 — φ Parameter Tuning on Sphere")
    print("=" * 60)

    cfg.cec_runs = 10           # fewer runs for quick tuning test
    phi_results  = tune_phi(sphere, cfg)
    plot_phi_comparison(phi_results)

    #PHASE 4 — CEC 2017 Full Benchmark
    #Uncomment when cec17.so is compiled and input_data/ is present
    print("\n" + "=" * 60)
    print("  PHASE 4 — CEC 2017 Full Benchmark")
    print("=" * 60)

    cfg.cec_runs = 30
    cfg.phi = 0.5        # best φ from Phase 3 tuning
    report = run_cec2017_benchmark(cfg)
    plot_benchmark_comparison(report)


if __name__ == "__main__":
    np.random.seed(42)
    main()