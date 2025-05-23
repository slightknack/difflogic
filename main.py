import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from pprint import pprint

def gate_all(a, b):
    return jnp.array([
        jnp.zeros_like(a),
        a * b,
        a - a*b,
        a,
        b - a*b,
        b,
        a + b - 2.0*a*b,
        a + b - a*b,
        1.0 - (a + b - a*b),
        1.0 - (a + b - 2.0*a*b),
        1.0 - b,
       	1.0 - b + a*b,
        1.0 - a,
        1.0 - a + a*b,
        1.0 - a*b,
        jnp.ones_like(a),
    ])

# gate_all(left, right) and w have shape (16, n)
# where n is the dimension of left/right
# This is a batched dot product along the second axis (axis 1)
def gate(left, right, w):
    return vmap(jnp.dot, in_axes=1)(gate_all(left, right), w)

def relu(left, right, w):
    return jnp.maximum(0., left)

# m rows by n columns
# n is input dim, m is output dim
def rand_weight_bias(key, m, n, scale=1e-2):
    w_key, b_key = random.split(key)
    w = scale * random.normal(w_key, (n, m))
    b = scale * random.normal(b_key, (n,))
    return (w, b)

def gate_normalize(w):
    sum_col = jnp.sum(w, axis=0)
    return w / sum_col[None,:]

# uniform random vectors length 16 whose entries sum to 1
def rand_gate(key, n):
    return gate_normalize(random.uniform(key, (16, n)))

def rand_layer(key, m, n):
    left_key, right_key, gate_key = random.split(key, 3)
    left = rand_weight_bias(left_key, m, n)
    right = rand_weight_bias(right_key, m, n)
    gate = rand_gate(gate_key, n)
    output = (*left, *right, gate)
    assert 5, len(output)
    return output

def rand_network(key, sizes):
    keys = random.split(key, len(sizes))
    dims = zip(keys, sizes[:-1], sizes[1:])
    return [rand_layer(*dim) for dim in dims]

def predict(params, inp):
    active = inp
    for (lw, lb, rw, rb, g) in params:
        outs_l = jnp.dot(lw, active) + lb
        outs_r = jnp.dot(rw, active) + rb
        active = relu(outs_l, outs_r, g)
    return active

predict_batch = vmap(predict, in_axes=(None, 0))

# l2 loss
def loss(params, inp, out):
    preds = predict_batch(params, inp)
    return jnp.mean(jnp.square(preds - out))

@jit
def update(params, x, y, step_size):
    grads = grad(loss)(params, x, y)
    return [(
        lw - step_size * dlw,
        lb - step_size * dlb,
        rw - step_size * drw,
        rb - step_size * drb,
        g - step_size * dg,
    ) for
        (lw, lb, rw, rb, g),
        (dlw, dlb, drw, drb, dg)
    in zip(params, grads)]

# def accuracy_batch(params, x, y):
#     y_hat = predict_batch(params, x)
#     return jnp.mean(jnp.square(y - y_hat))

def conway_kernel(inp):
    def c_and(a, b): return a * b
    def c_or(a, b): return a + b - a * b
    def c_eq(a, b): return jnp.maximum(0.0, 1.0 - jnp.abs(a - b)) # _/\_
    alive = inp[4]
    inp = inp.at[4].set(0)
    neighbors = jnp.sum(inp)
    return c_or(c_eq(3, neighbors), c_and(alive, c_eq(2, neighbors)))

conway_kernel_batch = lambda x: jnp.expand_dims(vmap(conway_kernel)(x), axis=-1)

# def conway_draw(inp):
#     out = conway_kernel(inp)
#     inp = inp.reshape((3, 3))
#     for row in inp:
#         for x in row:
#             print("x" if x > 0.5 else "-", end="")
#         print(" X" if out > 0.5 else " _")

def conway_sample(key):
    return jnp.round(random.uniform(key, (9,), maxval=0.67))

def conway_sample_batch(key, size):
    keys = random.split(key, size)
    return vmap(conway_sample)(keys)

def train_print_loss(key, params, x, y):
    x_test = conway_sample_batch(key, x.shape[0])
    y_test = conway_kernel_batch(x_test)
    train_loss = loss(params, x, y)
    test_loss = loss(params, x_test, y_test)
    print(f"train_loss: {train_loss:.3g}", end="; ")
    print(f"test_loss: {test_loss:.3g}")

def train(key, params, step_size=0.01, epochs=10000, batch_size=1024):
    import time
    keys = random.split(key, epochs)
    for (i, key_epoch) in enumerate(keys):
        time_start = time.time()
        key_train, key_accuracy = random.split(key_epoch)
        x = conway_sample_batch(key_train, batch_size)
        y = conway_kernel_batch(x)
        params = update(params, x, y, step_size)
        time_epoch = time.time() - time_start
        print(f"Epoch ({i+1}/{epochs}) in {time_epoch:.3g}s", end="; ")
        train_print_loss(key_accuracy, params, x, y)
    return params

if __name__ == "__main__":
    key = random.PRNGKey(379009)
    rand_key, train_key = random.split(key)

    layer_sizes = [9, 512, 512, 1]
    params = rand_network(key, layer_sizes)
    # pprint(params)

    train(key, params)
