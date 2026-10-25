"""MLP + training many networks in parallel on the GPU.

vmap = train many networks at once, jit = compile. vmap needs identical shapes for
all networks, so every network is built at max size and unused units / layers are masked.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.genome import SEARCH_SPACE

MAX_WIDTH = SEARCH_SPACE["width"][2]
SGD_MOMENTUM = 0.9
DATA_SEED = 1234      # image order - identical for every network
CPG = 24              # max networks per GPU call
GROUP = 8             # pad calls to a multiple of 8 networks -> fewer recompiles


def init_params(key, width):
    """Weights for one max-size network"""
    k = jax.random.split(key, 4)
    dense = lambda kk, n_in, n_out, fan_in: jax.random.normal(kk, (n_in, n_out)) / jnp.sqrt(fan_in)
    return dict(W1=dense(k[0], 784, MAX_WIDTH, 784), b1=jnp.zeros(MAX_WIDTH),
                W2=dense(k[1], MAX_WIDTH, MAX_WIDTH, width), b2=jnp.zeros(MAX_WIDTH),
                W3=dense(k[2], MAX_WIDTH, MAX_WIDTH, width), b3=jnp.zeros(MAX_WIDTH),
                Wo=dense(k[3], MAX_WIDTH, 10, width), bo=jnp.zeros(10))


def forward(p, cfg, x):
    """MLP with cfg['layers'] hidden layers of cfg['width'] units (the rest is masked out)."""
    mask = (jnp.arange(MAX_WIDTH) < cfg["width"]).astype(jnp.float32)
    act = lambda z: jnp.where(cfg["tanh"], jnp.tanh(z), jax.nn.relu(z))
    h = act(x @ p["W1"] + p["b1"]) * mask
    h = jnp.where(cfg["layers"] >= 2, act(h @ p["W2"] + p["b2"]) * mask, h)   # unused layer = skip
    h = jnp.where(cfg["layers"] >= 3, act(h @ p["W3"] + p["b3"]) * mask, h)
    return h @ p["Wo"] + p["bo"]


def accuracy(p, cfg, x, y):
    return jnp.mean(jnp.argmax(forward(p, cfg, x), axis=1) == y)


def train_net(cfg, key, batches, x_tr, y_tr, x_v, y_v, x_te, y_te):
    """Train ONE network. Returns validation accuracy after every chunk of batches + test accuracy."""
    params = init_params(key, cfg["width"])
    adam, sgd = optax.adam(cfg["lr"]), optax.sgd(cfg["lr"], momentum=SGD_MOMENTUM)
    opt_state = (adam.init(params), sgd.init(params))

    def loss(p, xb, yb):
        return optax.softmax_cross_entropy_with_integer_labels(forward(p, cfg, xb), yb).mean()

    def step(carry, idx):
        p, (s_adam, s_sgd) = carry
        g = jax.grad(loss)(p, x_tr[idx], y_tr[idx])
        u_adam, s_adam = adam.update(g, s_adam, p)
        u_sgd, s_sgd = sgd.update(g, s_sgd, p)
        # under vmap all networks run the same code: compute both updates, keep the chosen one
        u = jax.tree_util.tree_map(lambda a, b: jnp.where(cfg["adam"], a, b), u_adam, u_sgd)
        return (optax.apply_updates(p, u), (s_adam, s_sgd)), None

    def train_and_eval(carry, idx_chunk):
        carry, _ = jax.lax.scan(step, carry, idx_chunk)
        return carry, accuracy(carry[0], cfg, x_v, y_v)

    (params, _), curve = jax.lax.scan(train_and_eval, (params, opt_state), batches)
    return curve, accuracy(params, cfg, x_te, y_te)


# in_axes: 0 = differs per network, None = shared
train_many = jax.jit(jax.vmap(train_net, in_axes=(0, 0, None, None, None, None, None, None, None)))


def sample_order(n_samples, n_train, seed=DATA_SEED):
    """Order of training images: identical for every network. Reshuffles each epoch,
    so budgets larger than the training set work."""
    rng = np.random.default_rng(seed)
    n_epochs = -(-n_samples // n_train)
    return np.concatenate([rng.permutation(n_train) for _ in range(n_epochs)])[:n_samples]


def train_population(hps, data, budget, seed, eval_every=None):
    """Train a list of networks in parallel on `budget` training images.
    Returns final val accuracy (n,), learning curves (n, budget // eval_every), test accuracy (n,)."""
    every = eval_every or budget
    budget = (budget // every) * every
    order = sample_order(budget, len(data[0]))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(hps))
    curves, test = np.zeros((len(hps), budget // every)), np.zeros(len(hps))

    for bs in sorted({hp["batch_size"] for hp in hps}):     # different batch size = different shapes
        batches = jnp.asarray(order.reshape(budget // every, every // bs, bs))
        ids = [i for i, hp in enumerate(hps) if hp["batch_size"] == bs]
        for s in range(0, len(ids), CPG):
            part = ids[s:s + CPG]
            size = min(CPG, -(-len(part) // GROUP) * GROUP)
            padded = part + [part[-1]] * (size - len(part))
            cfg = dict(lr=jnp.array([hps[i]["lr"] for i in padded], dtype=jnp.float32),
                       adam=jnp.array([hps[i]["optimizer"] == "adam" for i in padded]),
                       layers=jnp.array([hps[i]["layers"] for i in padded]),
                       width=jnp.array([hps[i]["width"] for i in padded]),
                       tanh=jnp.array([hps[i]["activation"] == "tanh" for i in padded]))
            cu, te = train_many(cfg, keys[jnp.array(padded)], batches, *data)
            curves[part], test[part] = np.asarray(cu)[:len(part)], np.asarray(te)[:len(part)]
    return curves[:, -1], curves, test


def fitness(hps, data, budget, seed):
    """FITNESS = validation accuracy after `budget` training images, for each network."""
    return train_population(hps, data, budget, seed)[0]
