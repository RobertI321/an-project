"""The neural network (MLP) and training many of them in parallel on the GPU.

Trick: every network is built at the MAXIMUM size, and unused units / layers are masked out.
Then all networks have the same array shapes, so jax.vmap can train a whole group at once
(much faster than one by one), and jax.jit compiles it for the GPU.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax

SGD_MOMENTUM = 0.9
DATA_SEED = 1234      # order of the training images - identical for every network
GROUP_MAX = 24        # max networks per GPU call (limits GPU memory)
GROUP_PAD = 8         # pad groups to a multiple of 8 networks -> JAX recompiles less often


def init_params(key, width, max_width):
    """Random starting weights (scaled by 1/sqrt(inputs)) and zero biases, at max size."""
    keys = jax.random.split(key, 4)

    def weights(k, n_in, n_out, fan_in):
        return jax.random.normal(k, (n_in, n_out)) / jnp.sqrt(fan_in)

    return {
        "W1": weights(keys[0], 784, max_width, 784), "b1": jnp.zeros(max_width),
        "W2": weights(keys[1], max_width, max_width, width), "b2": jnp.zeros(max_width),
        "W3": weights(keys[2], max_width, max_width, width), "b3": jnp.zeros(max_width),
        "Wo": weights(keys[3], max_width, 10, width), "bo": jnp.zeros(10),
    }


def forward(params, net, images, max_width):
    """Images -> scores for the 10 classes. `net` holds this network's width / layers / activation."""
    mask = (jnp.arange(max_width) < net["width"]).astype(jnp.float32)    # 1 for used units, 0 for unused

    def activate(values):
        return jnp.where(net["tanh"], jnp.tanh(values), jax.nn.relu(values))

    hidden = activate(images @ params["W1"] + params["b1"]) * mask
    layer2 = activate(hidden @ params["W2"] + params["b2"]) * mask
    hidden = jnp.where(net["layers"] >= 2, layer2, hidden)              # skip layer 2 if the net has 1 layer
    layer3 = activate(hidden @ params["W3"] + params["b3"]) * mask
    hidden = jnp.where(net["layers"] >= 3, layer3, hidden)              # skip layer 3 if the net has < 3
    return hidden @ params["Wo"] + params["bo"]


def accuracy(params, net, images, labels, max_width):
    return jnp.mean(jnp.argmax(forward(params, net, images, max_width), axis=1) == labels)


def train_net(net, key, batches, train_x, train_y, val_x, val_y, test_x, test_y, max_width):
    """Train ONE network. batches: (n_evals, steps_per_eval, batch_size) image indices.
    Returns the learning curve (validation accuracy after every n_evals chunk) and the final test accuracy."""
    params = init_params(key, net["width"], max_width)
    adam = optax.adam(net["lr"])
    sgd = optax.sgd(net["lr"], momentum=SGD_MOMENTUM)

    def loss_fn(params, images, labels):
        scores = forward(params, net, images, max_width)
        return optax.softmax_cross_entropy_with_integer_labels(scores, labels).mean()

    def train_step(state, batch_indices):
        params, adam_state, sgd_state = state
        grads = jax.grad(loss_fn)(params, train_x[batch_indices], train_y[batch_indices])
        adam_updates, adam_state = adam.update(grads, adam_state, params)
        sgd_updates, sgd_state = sgd.update(grads, sgd_state, params)
        # under vmap every network runs the same code: compute both optimizers, keep the chosen one
        updates = jax.tree_util.tree_map(lambda a, s: jnp.where(net["adam"], a, s), adam_updates, sgd_updates)
        return (optax.apply_updates(params, updates), adam_state, sgd_state), None

    def train_chunk(state, chunk):
        state, _ = jax.lax.scan(train_step, state, chunk)
        return state, accuracy(state[0], net, val_x, val_y, max_width)

    state = (params, adam.init(params), sgd.init(params))
    state, curve = jax.lax.scan(train_chunk, state, batches)
    return curve, accuracy(state[0], net, test_x, test_y, max_width)


@functools.lru_cache
def compiled_trainer(max_width):
    """train_net for a whole group at once: vmap over networks (0) with shared data (None), compiled."""
    one_net = functools.partial(train_net, max_width=max_width)
    return jax.jit(jax.vmap(one_net, in_axes=(0, 0, None, None, None, None, None, None, None)))


def image_order(n_images, n_train, seed=DATA_SEED):
    """Which training image comes when - the same for every network. New shuffle every epoch."""
    rng = np.random.default_rng(seed)
    n_epochs = -(-n_images // n_train)
    return np.concatenate([rng.permutation(n_train) for _ in range(n_epochs)])[:n_images]


class Trainer:
    """Trains lists of networks on one dataset."""

    def __init__(self, data, max_width):
        self.data = data                       # (train_x, train_y, val_x, val_y, test_x, test_y)
        self.max_width = max_width
        self.n_train = len(data[0])

    def fitness(self, hps, budget, seed):
        """FITNESS = validation accuracy after `budget` training images, for each network."""
        curves, _ = self.train(hps, budget, seed)
        return curves[:, -1]

    def train(self, hps, n_images, seed, eval_every=None):
        """Train networks (list of hyperparameter dicts) from scratch on n_images training images.
        Returns learning curves (n_networks, n_images // eval_every) and test accuracies (n_networks,)."""
        eval_every = eval_every or n_images
        n_images = (n_images // eval_every) * eval_every
        order = image_order(n_images, self.n_train)
        keys = jax.random.split(jax.random.PRNGKey(seed), len(hps))      # one starting-weights key per network
        curves = np.zeros((len(hps), n_images // eval_every))
        test = np.zeros(len(hps))
        train_many = compiled_trainer(self.max_width)

        # networks with different batch sizes have different shapes -> train them in separate groups
        for batch_size in sorted({hp["batch_size"] for hp in hps}):
            batches = jnp.asarray(order.reshape(n_images // eval_every, eval_every // batch_size, batch_size))
            ids = [i for i, hp in enumerate(hps) if hp["batch_size"] == batch_size]
            for start in range(0, len(ids), GROUP_MAX):
                group = ids[start:start + GROUP_MAX]
                size = min(GROUP_MAX, -(-len(group) // GROUP_PAD) * GROUP_PAD)
                padded = group + [group[-1]] * (size - len(group))        # extra copies are thrown away
                net = {
                    "lr": jnp.array([hps[i]["lr"] for i in padded], dtype=jnp.float32),
                    "adam": jnp.array([hps[i]["optimizer"] == "adam" for i in padded]),
                    "layers": jnp.array([hps[i]["layers"] for i in padded]),
                    "width": jnp.array([hps[i]["width"] for i in padded]),
                    "tanh": jnp.array([hps[i]["activation"] == "tanh" for i in padded]),
                }
                group_curves, group_test = train_many(net, keys[jnp.array(padded)], batches, *self.data)
                curves[group] = np.asarray(group_curves)[:len(group)]
                test[group] = np.asarray(group_test)[:len(group)]
        return curves, test
