import random
import json
import time
import numpy as np
from deap import base, creator, tools, algorithms

from support import fitness, NR_TRANSITIONS, NR_PLACES

POP_SIZE   = 100
NGEN       = 50
LOW        = 0
UP         = NR_PLACES - 1
IND_SIZE   = NR_TRANSITIONS * 2
TOURN_SIZE = 2
INDPB      = 0.1


CX_PROBS  = [0.2, 0.4, 0.6, 0.8]
MUT_PROBS = [0.2, 0.4, 0.6, 0.8]
N_RUNS    = 10


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene",       random.randint, LOW, UP)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate",     tools.cxTwoPoint)
toolbox.register("mutate",   tools.mutUniformInt, low=LOW, up=UP, indpb=INDPB)
toolbox.register("select",   tools.selTournament, tournsize=TOURN_SIZE)


def run_trial(cxpb, mutpb):
    """Run the GA once with the given crossover and mutation probabilities.

    Returns the best fitness found and the generation-by-generation logbook.
    A fresh population is created each call so runs are independent.
    """
    pop = toolbox.population(n=POP_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("avg",  np.mean)

    hof = tools.HallOfFame(1)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cxpb, mutpb=mutpb,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    best_fitness = hof[0].fitness.values[0]
    return best_fitness, logbook


def run_experiment():
    # results[key]  = list of N_RUNS best-fitness floats
    # logbooks[key] = list of N_RUNS logbooks (each: list of gen records)
    results  = {}
    logbooks = {}

    total = len(CX_PROBS) * len(MUT_PROBS) * N_RUNS
    done  = 0

    for cxpb in CX_PROBS:
        for mutpb in MUT_PROBS:
            key = f"{cxpb}_{mutpb}"
            results[key]  = []
            logbooks[key] = []

            for run in range(1, N_RUNS + 1):
                best, logbook = run_trial(cxpb, mutpb)
                results[key].append(best)
                logbooks[key].append(logbook)

                done += 1
                print(f"  cxpb={cxpb:.1f}  mutpb={mutpb:.1f}  "
                      f"run {run}/{N_RUNS}  "
                      f"best={best:.4f}  "
                      f"({done}/{total} total)")

    return results, logbooks


def print_table(results):
    col_w = 10

    header = "Mut \\ CX".ljust(col_w) + "".join(
        f"{cx:.1f}".rjust(col_w) for cx in CX_PROBS
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # find global best to mark with *
    best_abf = -1
    best_key = None
    for cxpb in CX_PROBS:
        for mutpb in MUT_PROBS:
            abf = np.mean(results[f"{cxpb}_{mutpb}"])
            if abf > best_abf:
                best_abf = abf
                best_key = f"{cxpb}_{mutpb}"

    for mutpb in MUT_PROBS:
        row = f"{mutpb:.1f}".ljust(col_w)
        for cxpb in CX_PROBS:
            key = f"{cxpb}_{mutpb}"
            abf = np.mean(results[key])
            cell = f"{abf:.4f}"
            if key == best_key:
                cell += "*"
            row += cell.rjust(col_w)
        print(row)

    print("=" * len(header))
    best_cx, best_mut = best_key.split("_")
    print(f"\nBest: cxpb={best_cx}  mutpb={best_mut}  ABF={best_abf:.4f}")


def save_results(results, logbooks):
    # Build a serialisable summary
    summary = {
        "config": {
            "ngen":       NGEN,
            "pop_size":   POP_SIZE,
            "n_runs":     N_RUNS,
            "tournsize":  TOURN_SIZE,
            "indpb":      INDPB,
            "cx_ops":     "cxTwoPoint",
            "mut_op":     "mutUniformInt",
            "sel_op":     f"selTournament(tournsize={TOURN_SIZE})",
        },
        "results":  {},
        "logbooks": {},
        "best":     {},
    }

    best_abf = -1
    best_cx = best_mut = None

    for cxpb in CX_PROBS:
        for mutpb in MUT_PROBS:
            key = f"{cxpb}_{mutpb}"
            abf = float(np.mean(results[key]))
            summary["results"][key] = {
                "cxpb": cxpb,
                "mutpb": mutpb,
                "abf":  abf,
                "runs": [float(v) for v in results[key]],
            }
            # logbook: list of runs, each run = list of {gen, best, avg}
            summary["logbooks"][key] = [
                [{"gen": r["gen"], "best": float(r["best"]), "avg": float(r["avg"])}
                 for r in lb]
                for lb in logbooks[key]
            ]
            if abf > best_abf:
                best_abf = abf
                best_cx, best_mut = cxpb, mutpb

    summary["best"] = {"cxpb": best_cx, "mutpb": best_mut, "abf": best_abf}

    with open("task3_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to task3_results.json")


if __name__ == "__main__":
    print(f"Task 3 – Hyperparameter tuning")
    print(f"Grid: cxpb={CX_PROBS}  x  mutpb={MUT_PROBS}")
    print(f"Runs per combo: {N_RUNS} | Generations: {NGEN} | Pop: {POP_SIZE}\n")

    t0 = time.time()
    results, logbooks = run_experiment()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min\n")
    print_table(results)
    save_results(results, logbooks)
