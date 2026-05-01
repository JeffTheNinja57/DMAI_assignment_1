import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from support import fitness, NR_TRANSITIONS, NR_PLACES

# GA Parameters
POP_SIZE    = 100   # number of individuals in the population initial: 100
CX_PROB     = 0.7   # probability of crossover initial :0.7
MUT_PROB    = 0.2   # probability of mutation initial: 0.2
NGEN        = 50    # number of generations initial: 50
TOURN_SIZE  = 3     # tournament size for selection initial: 3
LOW         = 0     # minimum value of a gene 
UP          = NR_PLACES - 1  # maximum value of a gene (0–8) 
IND_SIZE    = NR_TRANSITIONS * 2  # chromosome length = 12 * 2

#  DEAP Setup 

# Tell DEAP to MAXIMIZE fitness (weight = +1.0)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Individual = list with a fitness attribute attached
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Single gene = random integer between 0 and 8
toolbox.register("gene", random.randint, LOW, UP)

# Individual = list of 24 random genes
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)

# Population = list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the operators
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=LOW, up=UP, indpb=0.1) # each gene has a 10% chance of being changed, controls how many genes get changed if theres mutation initially set to 0.1
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

# Run the GA

def run_ga():
    # create initial population
    pop = toolbox.population(n=POP_SIZE)

    # statistics to track best fitness each generation
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("avg", np.mean)

    # keeps track of the single best individual ever seen
    hof = tools.HallOfFame(1)

    
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    return pop, logbook, hof

# Plot Results

def plot_fitness(logbook):
    generations = logbook.select("gen")
    best_fitness = logbook.select("best") #pulls out the best fitness value recorded at each generation

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label="Best Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Fitness per Generation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("task1_fitness_plot.png")
    plt.show()
    print("Plot saved as task1_fitness_plot.png")


# Main

if __name__ == "__main__":
    print("Starting GA...")
    print(f"Population: {POP_SIZE} | Generations: {NGEN} | "
          f"CX: {CX_PROB} | MUT: {MUT_PROB} | Tournament: {TOURN_SIZE}")

    pop, logbook, hof = run_ga()

    print(f"\nBest fitness found: {hof[0].fitness.values[0]:.4f}")
    print(f"Best individual: {list(hof[0])}")

    plot_fitness(logbook)