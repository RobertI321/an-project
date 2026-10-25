"""Download the dataset once (cached in data/), scale to [0, 1], fixed 50k/10k train/val split."""
import gzip
import io
import os
import urllib.request

import jax.numpy as jnp
import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"


def _download(name):
    def get(f):
        for attempt in range(5):
            try:
                return urllib.request.urlopen(URL + f, timeout=30).read()
            except OSError as e:
                print(f"{f}: download failed ({e}), retrying...")
        raise RuntimeError(f"could not download {f}")

    if name == "mnist":
        d = np.load(io.BytesIO(get("mnist.npz")))
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    img = lambda f: np.frombuffer(gzip.decompress(get(f)), np.uint8, offset=16).reshape(-1, 784)
    lab = lambda f: np.frombuffer(gzip.decompress(get(f)), np.uint8, offset=8)
    return (img("train-images-idx3-ubyte.gz"), lab("train-labels-idx1-ubyte.gz"),
            img("t10k-images-idx3-ubyte.gz"), lab("t10k-labels-idx1-ubyte.gz"))

def load_data(name, n_train=50_000, split_seed=42):
    """Returns (x_train, y_train, x_val, y_val, x_test, y_test) as JAX arrays."""
    cache = f"data/{name}.npz"
    if not os.path.exists(cache):
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(cache, *_download(name))
    x, y, xt, yt = np.load(cache).values()

    idx = np.random.default_rng(split_seed).permutation(len(x))   # fixed 50k/10k split
    train, val = idx[:n_train], idx[n_train:]
    img = lambda a: jnp.asarray(a.reshape(len(a), 784).astype(np.float32) / 255.0)
    lab = lambda a: jnp.asarray(a.astype(np.int32))
    print(f"{name}: train {len(train)}, val {len(val)}, test {len(xt)}")
    return img(x[train]), lab(y[train]), img(x[val]), lab(y[val]), img(xt), lab(yt)
