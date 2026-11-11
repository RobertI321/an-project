"""The four search methods. Each one only needs two methods:
    propose(n)            -> n new genomes to try
    update(genomes, fit)  -> learn from how well they did
run_search() is the loop around them, the same for every method.
"""
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def mutate(genomes, rng, sigma=0.1):
    """Add a little random noise to every gene."""
    return np.clip(genomes + sigma * rng.normal(size=genomes.shape), 0.0, 1.0)


def crossover(a, b, rng):
    """Every gene comes from parent a or parent b with 50% chance."""
    return np.where(rng.random(a.shape) < 0.5, a, b)


class RandomSearch:
    """Control: random genomes every time, never learns."""
    def __init__(self, seed, space):
        self.rng, self.space = np.random.default_rng(seed), space

    def propose(self, n):
        return self.rng.random((n, self.space.n_genes))

    def update(self, genomes, fit):
        pass


class GA:
    """Genetic algorithm: better networks are more likely parents; children = crossover + mutation;
    the best of parents + children survive."""
    def __init__(self, seed, space):
        self.rng, self.space = np.random.default_rng(seed), space
        self.pop, self.fit = None, None             # current population and its fitness

    def propose(self, n):
        if self.pop is None:                        # first generation is random
            return self.rng.random((n, self.space.n_genes))
        weights = self.fit - self.fit.min() + 1e-6
        p = weights / weights.sum()                 # selection chance proportional to accuracy
        parents_a = self.pop[self.rng.choice(len(self.pop), n, p=p)]
        parents_b = self.pop[self.rng.choice(len(self.pop), n, p=p)]
        return mutate(crossover(parents_a, parents_b, self.rng), self.rng)

    def update(self, genomes, fit):
        if self.pop is None:
            self.pop, self.fit = np.array(genomes), np.array(fit)
            return
        all_genomes = np.concatenate([self.pop, genomes])      # parents compete with children
        all_fit = np.concatenate([self.fit, fit])
        keep = np.argsort(all_fit)[::-1][:len(genomes)]
        self.pop, self.fit = all_genomes[keep], all_fit[keep]


class MAPElites:
    """Keeps the best network ('elite') in every cell of the learning-rate x size grid,
    and breeds new networks from random elites -> stays diverse."""
    def __init__(self, seed, space):
        self.rng, self.space = np.random.default_rng(seed), space
        self.archive = {}                           # grid cell -> (genome, fitness)

    def propose(self, n):
        if not self.archive:                        # first generation is random
            return self.rng.random((n, self.space.n_genes))
        elites = np.array([g for g, f in self.archive.values()])
        parents_a = elites[self.rng.integers(len(elites), size=n)]
        parents_b = elites[self.rng.integers(len(elites), size=n)]
        return mutate(crossover(parents_a, parents_b, self.rng), self.rng)

    def update(self, genomes, fit):
        for genome, f in zip(genomes, fit):
            cell = self.space.grid_cell(self.space.decode(genome))
            if cell not in self.archive or f > self.archive[cell][1]:
                self.archive[cell] = (genome, f)


class OptunaTPE:
    """Model-based (Bayesian) optimizer, not evolutionary: suggests genomes its model thinks score well."""
    def __init__(self, seed, space):
        self.space = space
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.trials = []

    def propose(self, n):
        self.trials = [self.study.ask() for _ in range(n)]
        return np.array([[t.suggest_float(f"g{i}", 0.0, 1.0) for i in range(self.space.n_genes)]
                         for t in self.trials])

    def update(self, genomes, fit):
        for trial, f in zip(self.trials, fit):
            self.study.tell(trial, float(np.nan_to_num(f)))


SEARCH_METHODS = {"GA": GA, "MAP-Elites": MAPElites, "Optuna TPE": OptunaTPE, "Random search": RandomSearch}


def run_search(method, space, fitness_fn, cpg, generations):
    """One search: `generations` rounds of `cpg` networks.
    Returns every genome tried, its fitness, and the best fitness found so far after each round."""
    all_genomes, all_fitness, best_so_far = [], [], []
    for _ in range(generations):
        genomes = method.propose(cpg)                             # 1. pick candidates
        fit = fitness_fn([space.decode(g) for g in genomes])      # 2. train + measure
        method.update(genomes, fit)                               # 3. learn from the results
        all_genomes += list(genomes)
        all_fitness += list(fit)
        best_so_far.append(max(all_fitness))
    return np.array(all_genomes), np.array(all_fitness), np.array(best_so_far)
