import random
import time
import json 
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from support import fitness, sol_to_graphviz, NR_TRANSITIONS, NR_PLACES
from repair import repair, constraint_repair


# best settings from Tasks 2 and 3.
TOURN_SIZE = 2
INDPB      = 0.1   
CXPB       = 0.8
MUTPB      = 0.4

# run length parameters 
POP_SIZE   = 150
NGEN       = 500
SEED       = 42
FAST_DEBUG = False    # just to debug at beginning 


# DEAP creator classes: maximise fitness, individuals are lists with a fitness attributes also
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def build_toolbox(use_decoration):
    """Build a toolbox with the chosen settings, optionally attaches the
    constraint repair decorator into mate/mutate.
    """
    tb = base.Toolbox()

    # an individual = NR_TRANSITIONS (12) * 2 = 24 random integers in [0, 8]
    tb.register("gene", random.randint, 0, NR_PLACES - 1)
    # creates the individual from repeated genes
    tb.register("individual", tools.initRepeat, creator.Individual,
                tb.gene, n=NR_TRANSITIONS * 2)
    # creates the population from the individual 
    tb.register("population", tools.initRepeat, list, tb.individual)

    # the supplied PETRINAS fitness function
    tb.register("evaluate", fitness)

    # tasks 2 best results 
    tb.register("mate",     tools.cxTwoPoint)
    tb.register("mutate",   tools.mutUniformInt,
                low=0, up=NR_PLACES - 1, indpb=INDPB)
    tb.register("select",   tools.selTournament, tournsize=TOURN_SIZE)

    # Only used in the AFTER run. This wraps mate/mutate so every offspring
    # is passed through repair() before being returned
    if use_decoration:
        tb.decorate("mate",   constraint_repair())
        tb.decorate("mutate", constraint_repair())
    return tb


def run_ga(use_decoration, label, pop_size, ngen):
    """Run eaSimple once. Same seed is used in every call so the
    BEFORE and AFTER runs start from byte-identical populations.
    Returns (best_individual, logbook, seconds).
    """
    print(f"\n[{label}] decoration={use_decoration}  "
          f"pop={pop_size}  ngen={ngen}")

    # reset both RNGs so this run starts from the same state as the other
    random.seed(SEED)
    np.random.seed(SEED)

    tb = build_toolbox(use_decoration)
    pop = tb.population(n=pop_size)

    # repair gen 0 manually so every individual ever evaluated satisfies C3 and C5
    # as decorator only reparis offspring
    if use_decoration:
        for ind in pop:
            repair(ind)

    # track best/avg fitness per generation, keep the all-time best
    stats = tools.Statistics(lambda ind: ind.fitness.values) # pull out fitness vals for each ind
    stats.register("best", np.max) # per gen
    stats.register("avg",  np.mean) # per gen 
    hof = tools.HallOfFame(1) # best all time individual 

    t0 = time.time()
    pop, logbook = algorithms.eaSimple(
        pop, tb, cxpb=CXPB, mutpb=MUTPB, ngen=ngen,
        stats=stats, halloffame=hof, verbose=False,
    )
    elapsed = time.time() - t0

    print(f"[{label}] best={hof[0].fitness.values[0]:.4f}  "
          f"elapsed={elapsed/60:.1f} min")
    return hof[0], logbook, elapsed


def plot_curves(log_before, log_after, path="task5_fitness_curves.png"):
    """Plot best fitness per generation for both runs on one figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(log_before.select("gen"), log_before.select("best"),
             label="Before: no decoration", color="tab:red")
    plt.plot(log_after.select("gen"), log_after.select("best"),
             label="After: with decoration", color="tab:blue")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Task 5: best fitness per generation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# use small sizes if FAST_DEBUG is set, otherwise the real ones
pop_size, ngen = (30, 10) if FAST_DEBUG else (POP_SIZE, NGEN)

# before with no contraint repair
best_before, log_before, t1 = run_ga(False, "BEFORE", pop_size, ngen)

# after wirth constraint repair
best_after,  log_after,  t2 = run_ga(True,  "AFTER",  pop_size, ngen)

# write results to json file 
with open("task5_best.json", "w") as f:
    json.dump({"before": list(best_before), "after": list(best_after)}, f)

# render both Petri nets and the fitness-curve comparison
sol_to_graphviz(list(best_before), output_file="before.pdf")
sol_to_graphviz(list(best_after),  output_file="after.pdf")
plot_curves(log_before, log_after)

# print summary 
print("\nTask 5 summary")
print(f"  BEFORE  fitness = {best_before.fitness.values[0]:.4f}  ({t1/60:.1f} min)")
print(f"  AFTER   fitness = {best_after.fitness.values[0]:.4f}  ({t2/60:.1f} min)")
