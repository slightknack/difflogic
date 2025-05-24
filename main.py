import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
from pprint import pprint

GATES = 16

def gate_all(a, b):
    return jnp.array([
        # jnp.maximum(0., a),
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
def gate(left, right, w, hard):
    w_gate = \
        jnp.exp(w) / jnp.sum(jnp.exp(w), axis=0, keepdims=True) if not hard \
        else jax.nn.one_hot(jnp.argmax(w, axis=0), GATES).T
    return jnp.sum(gate_all(left, right) * w_gate, axis=0)

def relu(left, right, w):
    return jnp.maximum(0., left)

# m rows by n columns
# n is input dim, m is output dim
def rand_weight_bias(key, m, n, scale=1e-2):
    w_key, b_key = random.split(key)
    w = scale * random.normal(w_key, (n, m))
    b = scale * random.normal(b_key, (n,))
    return (w, b)

# uniform random vectors length 16 whose entries sum to 1
def rand_gate(_key, n):
    return jnp.full((GATES, n), 1  / GATES)

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

def predict(params, inp, hard):
    active = inp
    for (lw, lb, rw, rb, g) in params:
        outs_l = jnp.dot(lw, active) + lb
        outs_r = jnp.dot(rw, active) + rb
        active = relu(outs_l, outs_r, g, )#hard)
    return active

predict_batch = vmap(predict, in_axes=(None, 0, None))

# l2 loss
def loss(params, inp, out, hard):
    preds = predict_batch(params, inp, hard)
    return jnp.mean(jnp.square(preds - out))

@jit
def update_sgd(params, x, y, step_size):
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

# @jit
def update_adamw(params, x, y, opt, opt_state):
    grads = grad(loss)(params, x, y, False)
    grads, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, grads)
    return new_params, opt_state

def conway_kernel(inp):
    def c_and(a, b): return a * b
    def c_or(a, b): return a + b - a * b
    def c_eq(a, b): return jnp.maximum(0.0, 1.0 - jnp.abs(a - b)) # _/\_
    alive = inp[4]
    inp = inp.at[4].set(0)
    neighbors = jnp.sum(inp)
    return c_or(c_eq(3, neighbors), c_and(alive, c_eq(2, neighbors)))

conway_kernel_batch = lambda x: jnp.expand_dims(vmap(conway_kernel)(x), axis=-1)

def conway_draw(inp):
    out = conway_kernel(inp)
    inp = inp.reshape((3, 3))
    for row in inp:
        for x in row:
            print("x" if x > 0.5 else "-", end="")
        print(" X" if out > 0.5 else " _")

def conway_sample(key):
    return jnp.round(random.uniform(key, (9,), maxval=0.67))

def conway_sample_batch(key, size):
    keys = random.split(key, size)
    return vmap(conway_sample)(keys)

def train_adamw(key, params, epochs=2000, batch_size=20):
    import time
    keys = random.split(key, epochs)
    opt = optax.adamw(learning_rate=0.05, b1=0.9, b2=0.99, weight_decay=1e-2)
    opt_state = opt.init(params)
    for (i, key_epoch) in enumerate(keys):
        key_train, key_accuracy = random.split(key_epoch)

        time_start = time.time()
        x = conway_sample_batch(key_train, batch_size)
        y = conway_kernel_batch(x)
        params, opt_state = update_adamw(params, x, y, opt, opt_state)
        time_epoch = time.time() - time_start

        print(f"Epoch ({i+1}/{epochs}) in {time_epoch:.3g}s", end="   \r")
        if i % 100 == 0: debug_loss(key_accuracy, params, x, y)
    return params

def debug_loss(key, params, x, y):
    x_test = conway_sample_batch(key, x.shape[0])
    y_test = conway_kernel_batch(x_test)
    train_loss = loss(params, x, y, False)
    test_loss = loss(params, x_test, y_test, False)
    test_loss_hard = loss(params, x_test, y_test, True)
    preds = predict_batch(params, x_test, False)
    print(preds[0:5].flatten(), y_test[0:5].flatten())
    print(f"train_loss: {train_loss:.3g}", end="; ")
    print(f"test_loss: {test_loss:.3g}", end="; ")
    print(f"test_loss_hard: {test_loss_hard:.3g}")

def debug_params(params):
    for i, layer in enumerate(params):
        print("LAYER", i)
        for j, param in enumerate(layer):
            print("PARAM", j)
            for vec in param.tolist():
                if isinstance(vec, list):
                    print("> ", end="")
                    for item in vec: print(f"{item:.3g} ", end="")
                    print()
                else:
                    print(f"{vec:.3g}")

if __name__ == "__main__":
    key = random.PRNGKey(379009)
    param_key, train_key = random.split(key)

    layer_sizes = [9, *([48] * 2), 1]
    params = rand_network(param_key, layer_sizes)

    params_trained = train_adamw(train_key, params)
    debug_params(params_trained)
