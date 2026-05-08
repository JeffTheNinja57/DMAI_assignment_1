import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from support import fitness, NR_TRANSITIONS, NR_PLACES

# GA Parameters
POP_SIZE  = 100              # population size (same as task 1 for fair comparison)
CX_PROB   = 0.7              # crossover probability
MUT_PROB  = 0.2              # mutation probability
NGEN      = 50               # generations per run
NR_RUNS   = 10               # independent runs per variant
LOW       = 0                # minimum gene value
UP        = NR_PLACES - 1   # maximum gene value (0–8)
IND_SIZE  = NR_TRANSITIONS * 2  # chromosome length = 12 × 2 = 24

# indpb for per-gene operators: 10% chance each gene is affected
INDPB_MUT = 0.1
# indpb for cxUniform: 50% chance each gene is swapped between parents (standard)
INDPB_CX  = 0.5

# DEAP Types (created once at module level)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Variant Definitions
# Each dict fully describes one operator combination to test.
# label  : short code used in plot axes and the summary table
# cx     : crossover operator name
# mut    : mutation operator name
# tourn  : tournament size for selTournament
VARIANTS = [
    {"label": "CxTP/mUI/T2",  "cx": "cxTwoPoint", "mut": "mutUniformInt",    "tourn": 2},
    {"label": "CxTP/mUI/T5",  "cx": "cxTwoPoint", "mut": "mutUniformInt",    "tourn": 5},
    {"label": "CxTP/mSI/T2",  "cx": "cxTwoPoint", "mut": "mutShuffleIndexes","tourn": 2},
    {"label": "CxTP/mSI/T5",  "cx": "cxTwoPoint", "mut": "mutShuffleIndexes","tourn": 5},
    {"label": "CxU/mUI/T2",   "cx": "cxUniform",  "mut": "mutUniformInt",    "tourn": 2},
    {"label": "CxU/mUI/T5",   "cx": "cxUniform",  "mut": "mutUniformInt",    "tourn": 5},
    {"label": "CxU/mSI/T2",   "cx": "cxUniform",  "mut": "mutShuffleIndexes","tourn": 2},
    {"label": "CxU/mSI/T5",   "cx": "cxUniform",  "mut": "mutShuffleIndexes","tourn": 5},
]
# Label legend: CxTP=cxTwoPoint, CxU=cxUniform, mUI=mutUniformInt,
#               mSI=mutShuffleIndexes, T2/T5=tournament size

# Toolbox Factory

def make_toolbox(cx, mut, tourn):
    """Return a fresh DEAP toolbox configured for the given operator combination."""
    tb = base.Toolbox()

    # Gene initialisation
    tb.register("gene", random.randint, LOW, UP)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.gene, n=IND_SIZE)
    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", fitness)

    # Crossover
    if cx == "cxTwoPoint":
        tb.register("mate", tools.cxTwoPoint)
    elif cx == "cxUniform":
        # indpb=0.5: each gene independently swapped with 50% probability
        tb.register("mate", tools.cxUniform, indpb=INDPB_CX)

    # Mutation
    if mut == "mutUniformInt":
        # indpb=0.1: each gene has 10% chance of being replaced by a random int
        tb.register("mutate", tools.mutUniformInt, low=LOW, up=UP, indpb=INDPB_MUT)
    elif mut == "mutShuffleIndexes":
        # indpb=0.1: each gene has 10% chance of being included in the shuffle
        tb.register("mutate", tools.mutShuffleIndexes, indpb=INDPB_MUT)

    # Selection
    tb.register("select", tools.selTournament, tournsize=tourn)

    return tb

# Single Run

def run_single(tb):
    """Run one GA with the given toolbox. Returns (best_per_gen list, final best fitness)."""
    pop = tb.population(n=POP_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)

    # HallOfFame keeps the single best individual seen across all generations
    hof = tools.HallOfFame(1)

    pop, logbook = algorithms.eaSimple(
        pop, tb,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=False   # silence per-generation output for cleaner multi-run logs
    )

    best_per_gen = logbook.select("best")
    return best_per_gen, hof[0].fitness.values[0]

# Run All Variants

def run_all_variants():
    """
    Run every variant NR_RUNS times.
    Returns a dict keyed by variant label:
        "all_best_per_gen" : list of NR_RUNS lists (best fitness at each generation)
        "best_finals"      : list of NR_RUNS final best fitness values
        "abf"              : average best fitness across runs (scalar)
        "time_s"           : total wall-clock time for all runs of this variant
    """
    results = {}

    for v in VARIANTS:
        label = v["label"]
        print(f"\n── Variant: {label} ──")
        tb = make_toolbox(v["cx"], v["mut"], v["tourn"])

        all_best_per_gen = []
        best_finals      = []

        t_start = time.time()
        run_times = []
        for run in range(NR_RUNS):
            t_run = time.time()
            best_per_gen, final_best = run_single(tb)
            run_elapsed = time.time() - t_run
            run_times.append(run_elapsed)

            all_best_per_gen.append(best_per_gen)
            best_finals.append(final_best)

            # estimate remaining time across ALL remaining runs in this variant
            avg_run = np.mean(run_times)
            runs_left = NR_RUNS - (run + 1)
            eta_s = avg_run * runs_left
            print(f"  run {run + 1:2d}/{NR_RUNS}  best={final_best:.4f}"
                  f"  run_time={run_elapsed:.1f}s  eta={eta_s:.0f}s")

        elapsed = time.time() - t_start

        results[label] = {
            "all_best_per_gen": all_best_per_gen,
            "best_finals":      best_finals,
            "abf":              float(np.mean(best_finals)),
            "time_s":           elapsed,
        }
        print(f"  ABF={results[label]['abf']:.4f}  total time={elapsed:.1f}s")

    return results

# Plotting

def plot_boxplot(results):
    """Boxplot of final best-fitness distribution (10 runs) for each variant."""
    labels      = list(results.keys())
    best_finals = [results[l]["best_finals"] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(best_finals, tick_labels=labels)
    ax.set_xlabel("Variant  (CxTP=cxTwoPoint, CxU=cxUniform, mUI=mutUniformInt, mSI=mutShuffleIndexes)")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Distribution of Best Fitness Across 10 Runs per Variant")
    ax.grid(True, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "task2_boxplot.png")
    plt.savefig(out)
    plt.show()
    print(f"Boxplot saved as {out}")


def plot_lineplot(results):
    """Line plot of Average Best Fitness (ABF) per generation for each variant."""
    generations = list(range(NGEN + 1))  # gen 0 (initial pop) through NGEN

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, data in results.items():
        # mean best fitness at each generation across all NR_RUNS runs
        abf_per_gen = np.mean(data["all_best_per_gen"], axis=0)
        ax.plot(generations, abf_per_gen, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Best Fitness (ABF)")
    ax.set_title("Average Best Fitness per Generation over 10 Runs")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "task2_lineplot.png")
    plt.savefig(out)
    plt.show()
    print(f"Line plot saved as {out}")

# Summary Table

def print_table(results):
    """Print the summary table required by the assignment."""
    header = f"{'Variant':<20} {'ABF':>8} {'Time (s)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for label, data in results.items():
        print(f"{label:<20} {data['abf']:>8.4f} {data['time_s']:>10.1f}")
    print("=" * len(header))

    best_label = max(results, key=lambda l: results[l]["abf"])
    print(f"\nBest variant by ABF: {best_label}  (ABF={results[best_label]['abf']:.4f})")

# Main 

if __name__ == "__main__":
    print("Task 2: Operator Comparison")
    print(f"Variants: {len(VARIANTS)} | Runs per variant: {NR_RUNS} | "
          f"Generations: {NGEN} | Population: {POP_SIZE}")
    print(f"CX prob: {CX_PROB} | MUT prob: {MUT_PROB} | "
          f"indpb_mut: {INDPB_MUT} | indpb_cx: {INDPB_CX}")

    results = run_all_variants()

    print_table(results)
    plot_boxplot(results)
    plot_lineplot(results)
