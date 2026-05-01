import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional


#CONFIGURATION
@dataclass
class PSOConfig:

    NP: int   = 30          #Number of particles
    dim: int  = 10          #Problem dimensionality
    T: int    = 500         #Max iterations

    #Search space
    lb: float = -100.0      #Lower bound
    ub: float =  100.0      #Upper bound

    #Standard PSO coefficients
    w:  float = 0.9         #Inertia weight (will decrease linearly to w_min)
    w_min: float = 0.4      #Minimum inertia weight
    c1: float = 2.0         #Cognitive coefficient
    c2: float = 2.0         #Social coefficient

    #Archive (Top-M)
    p: float  = 0.1         #Archive fraction  →  M = floor(p * NP)

    #phi parameter
    phi: float = 0.5        #Blending parameter φ ∈ [0, 1]

    #CEC 2017
    cec_func_id: int = 1    #CEC 2017 function ID (1–29)
    cec_runs: int    = 30   #Independent runs for statistics

    @property
    def M(self) -> int:
        return max(1, int(np.floor(self.p * self.NP)))


#PARTICLE

@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    pbest_pos: np.ndarray
    pbest_fit: float = np.inf   # Assuming minimisation

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


#STANDARD PSO

def run_standard_pso(func: Callable,
                     cfg:  PSOConfig) -> Tuple[np.ndarray, float, List[float]]:
    particles = initialise_swarm(cfg)
    fitnesses = evaluate_swarm(particles, func)

    #Initialise pbest
    for p, f in zip(particles, fitnesses):
        p.pbest_fit = f
        p.pbest_pos = p.position.copy()

    gbest_pos, gbest_fit = get_gbest(particles)
    curve = []

    for t in range(cfg.T):

        #Linearly decrease inertia weight
        w = cfg.w - (cfg.w - cfg.w_min) * (t / cfg.T)

        for p in particles:
            r1 = np.random.rand(cfg.dim)
            r2 = np.random.rand(cfg.dim)
            raise NotImplementedError("Phase 1: implement velocity update")
            raise NotImplementedError("Phase 1: implement position update")

            #Evaluate & update pbest
            fit = func(p.position)
            p.update_pbest(fit)

        #Update gbest
        gbest_pos, gbest_fit = get_gbest(particles)
        curve.append(gbest_fit)

    return gbest_pos, gbest_fit, curve


#ELITE ARCHIVE (TOP-M)

def build_archive(particles: List[Particle],
                  M: int) -> Tuple[List[Particle], float]:
    raise NotImplementedError("Phase 2: implement build_archive")


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

        #Build elite archive
        archive, f_worst_M = build_archive(particles, cfg.M)

        for p in particles:

            fit_current = func(p.position)

            #Phase 2: 1st Case
            raise NotImplementedError("Phase 2: implement Case ① check")

            #Phase 3: 2nd Case 
            raise NotImplementedError("Phase 3: implement φ-blended update")

            fit_new = func(p.position)
            p.update_pbest(fit_new)

        gbest_pos, gbest_fit = get_gbest(particles)
        curve.append(gbest_fit)

    return gbest_pos, gbest_fit, curve

#φ PARAMETER TUNING

def tune_phi(func: Callable,
             cfg:  PSOConfig,
             phi_values: Optional[List[float]] = None
             ) -> dict:
    if phi_values is None:
        phi_values = [round(v, 1) for v in np.arange(0.1, 1.0, 0.1)]

    results = {}

    for phi in phi_values:
        cfg.phi = phi
        run_results = []

        for _ in range(cfg.cec_runs):
            raise NotImplementedError("Phase 3: implement phi tuning loop")

        results[phi] = run_results
        print(f"  φ = {phi:.1f}  |  mean = {np.mean(run_results):.4e}"
              f"  |  std = {np.std(run_results):.4e}")

    return results

#CEC 2017 BENCHMARK EVALUATION

def load_cec2017_function(func_id: int, dim: int) -> Callable:
    raise NotImplementedError("Phase 4: plug in your CEC 2017 function loader")


def run_cec2017_benchmark(cfg: PSOConfig) -> dict:
    report = {}

    for func_id in range(1, 30):   # CEC 2017 has 29 functions
        cfg.cec_func_id = func_id
        func = load_cec2017_function(func_id, cfg.dim)

        std_results  = []
        mod_results  = []

        print(f"\n── CEC2017 F{func_id:02d} ──")

        for run in range(cfg.cec_runs):
            raise NotImplementedError("Phase 4: implement benchmark runs")

        report[func_id] = {
            'standard': {
                'mean': np.mean(std_results),
                'std':  np.std(std_results),
                'all_runs': std_results
            },
            'modified': {
                'mean': np.mean(mod_results),
                'std':  np.std(mod_results),
                'all_runs': mod_results
            }
        }

        print(f"  Standard PSO  |  mean={np.mean(std_results):.4e}  std={np.std(std_results):.4e}")
        print(f"  Modified PSO  |  mean={np.mean(mod_results):.4e}  std={np.std(mod_results):.4e}")

    return report


#PLOTTING UTILITIES

def plot_convergence(curves: dict, title: str = "Convergence Curve") -> None:
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
    plt.show()


def plot_phi_comparison(results: dict) -> None:
    labels = [f"φ={k}" for k in results.keys()]
    data   = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels)
    plt.yscale("log")
    plt.xlabel("φ value")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Modified PSO: φ Parameter Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_benchmark_comparison(report: dict) -> None:
    func_ids = list(report.keys())
    std_means = [report[f]['standard']['mean'] for f in func_ids]
    mod_means = [report[f]['modified']['mean']  for f in func_ids]

    x = np.arange(len(func_ids))
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
    plt.show()


#simple sample test function

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function — slightly harder test for Phase 1."""
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


#MAIN

def main():
    cfg = PSOConfig(NP=30, dim=10, T=500)

    print("=" * 60)
    print("  PHASE 1 — Standard PSO on Sphere function")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  PHASE 2 — Modified PSO (Elite Archive) on Sphere")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  PHASE 3 — φ Parameter Tuning on Sphere")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  PHASE 4 — CEC 2017 Full Benchmark")
    print("=" * 60)

    print("\nAll phases stubbed. Uncomment sections as you implement each phase.")


if __name__ == "__main__":
    np.random.seed(42)
    main()