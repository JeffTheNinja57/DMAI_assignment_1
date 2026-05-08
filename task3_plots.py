import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_boxplot(results):
    """Boxplot of final best-fitness distribution (10 runs) for each configuration."""
    # Sort the labels more or less chronologically (0.2_0.2, 0.2_0.4, etc.)
    labels = sorted(results.keys())
    best_finals = [results[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(best_finals, tick_labels=labels)
    ax.set_xlabel("Hyper-parameters (CrossoverProb_MutationProb)")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Distribution of Best Fitness Across 10 Runs per Configuration")
    ax.grid(True, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "task3_boxplot.png")
    plt.savefig(out)
    plt.show()


def plot_lineplot(logbooks_for_lineplot, ngen): 
    """Line plot of Average Best Fitness (ABF) per generation for each configuration."""
    generations = list(range(ngen + 1))   # gen 0 (initial pop) through NGEN
    labels = sorted(logbooks_for_lineplot.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        abf_per_gen = np.mean(logbooks_for_lineplot[label], axis=0)
        ax.plot(generations, abf_per_gen, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Best Fitness (ABF)")
    ax.set_title("Average Best Fitness per Generation over 10 Runs")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "task3_lineplot.png")
    plt.savefig(out) 
    plt.show()

if __name__ == "__main__":
    # Using the json file from the task3.py output
    with open("task3_results.json", "r") as file:
        data = json.load(file)
    
    # Create two simple dictionaries for the box - and line plots
    # Results for the box plot from the json
    results_for_boxplot = {k: v["runs"] for k, v in data["results"].items()}

    # Logbooks for the lineplot from the json
    logbooks_for_lineplot = {}
    for label, runs_list in data["logbooks"].items():
        all_runs_per_label = []
        for run in runs_list:
            best = []
            for gen_data in run:
                number = gen_data["best"]
                best.append(number)
            all_runs_per_label.append(best)
        logbooks_for_lineplot[label] = all_runs_per_label
    
    ngen_val = data["config"]["ngen"]

    plot_boxplot(results_for_boxplot)
    plot_lineplot(logbooks_for_lineplot, ngen_val)