"""The search space: which hyperparameters are searched and their ranges (from config.yaml).

A genome is one number in [0, 1] per gene. decode() turns it into real hyperparameters:
    log    -> e.g. learning rate 0.0001 .. 1, spread evenly on a log scale
    int    -> e.g. width 16 .. 256
    choice -> e.g. optimizer sgd / adam: [0, 1] is cut into equal slices, one per option
Search methods only ever work with the numbers; training only sees the decoded values.
"""
import numpy as np

GENES = ["lr", "optimizer", "batch_size", "layers", "width", "activation"]   # what train.py can use
MAX_LAYERS = 3                                                                # the model has max 3 hidden layers


class SearchSpace:
    def __init__(self, spec, grid_bins=8):
        if list(spec) != GENES:
            raise ValueError(f"search_space must list exactly these genes, in this order: {GENES}")
        if max(spec["layers"]["options"]) > MAX_LAYERS:
            raise ValueError(f"layers can be at most {MAX_LAYERS} (that's all the model supports)")
        self.spec = spec
        self.n_genes = len(spec)
        self.max_width = spec["width"]["max"]
        self.grid_bins = grid_bins

    def decode(self, genome):
        """Genome (numbers in [0, 1]) -> hyperparameters, e.g. {'lr': 0.003, 'optimizer': 'adam', ...}"""
        genome = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0)
        hp = {}
        for x, (name, gene) in zip(genome, self.spec.items()):
            if gene["type"] == "log":
                lo, hi = np.log10(gene["min"]), np.log10(gene["max"])
                hp[name] = float(10 ** (lo + x * (hi - lo)))
            elif gene["type"] == "int":
                hp[name] = int(round(gene["min"] + x * (gene["max"] - gene["min"])))
            else:
                options = gene["options"]
                hp[name] = options[min(int(x * len(options)), len(options) - 1)]
        return hp

    def encode(self, hp):
        """Hyperparameters -> genome (the reverse of decode)."""
        genome = []
        for name, gene in self.spec.items():
            value = hp[name]
            if gene["type"] == "log":
                lo, hi = np.log10(gene["min"]), np.log10(gene["max"])
                genome.append((np.log10(value) - lo) / (hi - lo))
            elif gene["type"] == "int":
                genome.append((value - gene["min"]) / (gene["max"] - gene["min"]))
            else:
                genome.append((gene["options"].index(value) + 0.5) / len(gene["options"]))
        return np.array(genome)

    def grid_cell(self, hp):
        """(row, col) of the diversity grid: row = learning rate, col = network size (both log scale)."""
        def position(value, lo, hi):          # 0..1 on a log scale
            return (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))

        lr, width, layers = self.spec["lr"], self.spec["width"], self.spec["layers"]
        smallest = n_params(dict(width=width["min"], layers=min(layers["options"])))
        largest = n_params(dict(width=width["max"], layers=max(layers["options"])))
        to_bin = lambda p: int(np.clip(p * self.grid_bins, 0, self.grid_bins - 1))
        return to_bin(position(hp["lr"], lr["min"], lr["max"])), to_bin(position(n_params(hp), smallest, largest))


def n_params(hp, n_in=784, n_out=10):
    """Number of weights in the network: input layer + extra hidden layers + output layer."""
    w, layers = hp["width"], hp["layers"]
    return n_in * w + w + (layers - 1) * (w * w + w) + w * n_out + n_out


def describe(hp):
    """One-line summary, e.g. 'adam lr=0.0030 batch= 64 2x165 tanh'."""
    return (f"{hp['optimizer']:<4} lr={hp['lr']:.4f} batch={hp['batch_size']:3d} "
            f"{hp['layers']}x{hp['width']:3d} {hp['activation']}")
