"""A genome is N_GENES numbers in [0, 1]. decode() turns it into the hyperparameters of one network."""
import numpy as np

# Search space - genes and their ranges (order = order of the genes in a genome)
SEARCH_SPACE = {
    "lr": ("log", 1e-4, 1.0),
    "optimizer": ("choice", ["sgd", "adam"]),
    "batch_size": ("choice", [32, 64, 128]),
    "layers": ("choice", [1, 2, 3]),        # number of hidden layers
    "width": ("int", 16, 256),              # units per hidden layer
    "activation": ("choice", ["relu", "tanh"]),
}
N_GENES = len(SEARCH_SPACE)
GRID_BINS = 8                               # MAP-Elites / diversity grid: learning rate x network size


def decode(genome):
    """Genome -> hyperparameters"""
    g = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0)
    hp = {}
    for x, (name, spec) in zip(g, SEARCH_SPACE.items()):
        kind = spec[0]
        if kind == "log":
            lo, hi = np.log10(spec[1]), np.log10(spec[2])
            hp[name] = float(10 ** (lo + x * (hi - lo)))
        elif kind == "int":
            hp[name] = int(round(spec[1] + x * (spec[2] - spec[1])))
        else:
            options = spec[1]
            hp[name] = options[min(int(x * len(options)), len(options) - 1)]
    return hp


def encode(hp):
    """Hyperparameters -> genome"""
    g = []
    for name, (kind, *spec) in SEARCH_SPACE.items():
        v = hp[name]
        if kind == "log":
            lo, hi = np.log10(spec[0]), np.log10(spec[1])
            g.append((np.log10(v) - lo) / (hi - lo))
        elif kind == "int":
            g.append((v - spec[0]) / (spec[1] - spec[0]))
        else:
            g.append((spec[0].index(v) + 0.5) / len(spec[0]))
    return np.array(g)


def describe(hp):
    return (f"{hp['optimizer']:<4} lr={hp['lr']:.4f} batch={hp['batch_size']:3d} "
            f"{hp['layers']}x{hp['width']:3d} {hp['activation']}")


def n_params(hp, n_in=784, n_out=10):
    w, L = hp["width"], hp["layers"]
    return n_in * w + w + (L - 1) * (w * w + w) + w * n_out + n_out


_P_MIN = n_params(dict(width=16, layers=1))
_P_MAX = n_params(dict(width=256, layers=3))


def grid_cell(hp):
    """(row, col) cell of the diversity grid for one network."""
    lr_pos = (np.log10(hp["lr"]) + 4) / 4                                    # 0..1
    size_pos = (np.log10(n_params(hp)) - np.log10(_P_MIN)) / (np.log10(_P_MAX) - np.log10(_P_MIN))
    to_bin = lambda p: int(np.clip(p * GRID_BINS, 0, GRID_BINS - 1))
    return to_bin(lr_pos), to_bin(size_pos)
