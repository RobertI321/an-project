"""The four search methods. All share the same two methods:
    propose(n)            -> n new genomes to try
    update(genomes, fit)  -> learn from their fitness
run_search() is the shared loop around them."""
import numpy as np
import optuna

from src.genome import N_GENES, decode, grid_cell

optuna.logging.set_verbosity(optuna.logging.WARNING)


def mutate(genomes, rng, sigma=0.1):
    """Add small random noise to every gene"""
    return np.clip(genomes + sigma * rng.normal(size=genomes.shape), 0.0, 1.0)


def crossover(a, b, rng):
    """Uniform crossover: every gene comes from parent a or parent b with 50% chance."""
    return np.where(rng.random(a.shape) < 0.5, a, b)


class RandomSearch:
    """Baseline search: random genomes."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def propose(self, n):
        return self.rng.random((n, N_GENES))

    def update(self, genomes, fit):
        pass


class GA:
    """Genetic algorithm: parent selection based on accuracy, crossover and mutation."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.pop, self.fit = None, None      # current population and its fitness

    def propose(self, n):
        if self.pop is None:                  # first generation is random
            return self.rng.random((n, N_GENES))
        w = self.fit - self.fit.min() + 1e-6  # accuracy-based selection
        p = w / w.sum()
        parents_a = self.pop[self.rng.choice(len(self.pop), n, p=p)]
        parents_b = self.pop[self.rng.choice(len(self.pop), n, p=p)]
        return mutate(crossover(parents_a, parents_b, self.rng), self.rng)

    def update(self, genomes, fit):
        if self.pop is None:
            self.pop, self.fit = np.array(genomes), np.array(fit)
            return
        all_g = np.concatenate([self.pop, genomes])   # parents compete with children
        all_f = np.concatenate([self.fit, fit])
        keep = np.argsort(all_f)[::-1][:len(genomes)]  # keep the best
        self.pop, self.fit = all_g[keep], all_f[keep]


class MAPElites:
    """Keeps the best network in every grid cell. New candidates are mutated children of random elites."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.archive = {}                     # cell -> (genome, fitness)

    def propose(self, n):
        if not self.archive:                  # first generation is random
            return self.rng.random((n, N_GENES))
        elites = np.array([g for g, f in self.archive.values()])
        parents_a = elites[self.rng.integers(len(elites), size=n)]
        parents_b = elites[self.rng.integers(len(elites), size=n)]
        return mutate(crossover(parents_a, parents_b, self.rng), self.rng)

    def update(self, genomes, fit):
        for g, f in zip(genomes, fit):
            cell = grid_cell(decode(g))
            if cell not in self.archive or f > self.archive[cell][1]:
                self.archive[cell] = (g, f)


class OptunaTPE:
    """A model-based optimizer. Suggests genomes that its model thinks will score well."""
    def __init__(self, seed):
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.trials = []

    def propose(self, n):
        self.trials = [self.study.ask() for _ in range(n)]
        return np.array([[t.suggest_float(f"g{i}", 0.0, 1.0) for i in range(N_GENES)] for t in self.trials])

    def update(self, genomes, fit):
        for t, f in zip(self.trials, fit):
            self.study.tell(t, float(np.nan_to_num(f)))


SEARCH_CLASSES = {"GA": GA, "MAP-Elites": MAPElites, "Optuna TPE": OptunaTPE, "Random search": RandomSearch}


def run_search(method, fit_fn, cpg, generations):
    """Shared loop for all four methods. fit_fn(list of hps) -> fitness array.
    Returns all genomes tried, their fitness, and the best fitness so far after each generation."""
    all_genomes, all_fitness, best = [], [], []
    for _ in range(generations):
        genomes = method.propose(cpg)                      # pick candidates
        fit = fit_fn([decode(g) for g in genomes])          # train + measure
        method.update(genomes, fit)                         # learn from results
        all_genomes += list(genomes)
        all_fitness += list(fit)
        best.append(max(all_fitness))
    return np.array(all_genomes), np.array(all_fitness), np.array(best)
