import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
from pprint import pprint
import functools
import itertools

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

# def relu(left, right, w):
#     return jnp.maximum(0., left)

def gate_normalize(w):
    sum_col = jnp.sum(w, axis=0)
    return w / sum_col[None,:]

# uniform random vectors length 16 whose entries sum to 1
def rand_gate(key, n):
    # return gate_normalize(random.uniform(key, (GATES, n)))
    result = jnp.zeros((GATES, n))
    result = result.at[3, :].set(10.0)
    return result
    # return jnp.full((GATES, n), 1. / GATES)

def pairs_rand(key, m, n_left):
    keys = random.split(key, n_left)
    return jnp.stack([random.permutation(key, m)[:2] for key in keys]).T

def pairs_rand_pad(key, pairs, m, n):
    if len(pairs) < n:
        print("↳ with random padding")
        pairs_pad = pairs_rand(key, m, n - len(pairs)).T
        pairs = jnp.concatenate([pairs, pairs_pad], axis=0)
    else:
        pairs = pairs[:n]
    return pairs

def wire_from_pairs(pairs, m, n):
    assert n, len(pairs)
    left = jax.nn.one_hot(pairs[0, :], num_classes=m)
    right = jax.nn.one_hot(pairs[1, :], num_classes=m)
    return left, right

def wire_rand(key, m, n):
    pairs = pairs_rand(key, m, n)
    return wire_from_pairs(pairs, m, n)

def wire_rand_unique(key, m, n):
    print("unique wire", m, "→", n)
    evens = zip(range(0, m)[0::2], range(0, m)[1::2])
    odds = zip(range(0, m)[1::2], list(range(0, m)[2::2]) + [0])
    pairs = jnp.array([*evens, *odds])
    key_rand, key_perm = random.split(key)
    pairs = pairs_rand_pad(key_rand, pairs, m, n)
    pairs = random.permutation(key_perm, pairs)
    return wire_from_pairs(pairs.T, m, n)

# m input, n output
def wire_rand_comb(key, m, n):
    print("comb wire", m, "→", n)
    key, key_sub = random.split(key)
    pairs = random.permutation(key_sub, jnp.array(list(itertools.combinations(list(range(m)), 2))))
    pairs = pairs_rand_pad(key, pairs, m, n)
    return wire_from_pairs(pairs.T, m, n)

def wire_tree(m, n):
    print("tree wire", m, "→", n)
    assert m, 2*n
    pairs = jnp.arange(m).reshape((2, n))
    return wire_from_pairs(pairs, m, n)

def rand_layer(key, m, n):
    left_key, right_key, gate_key = random.split(key, 3)
    # left, right = wire_tree(m, n) if m == 2*n \
    #     else wire_rand_unique(left_key, m, n)
    # left, right = wire_rand_comb(left_key, m, n)
    left, right = wire_rand_unique(left_key, m, n)
    param = rand_gate(gate_key, n)
    wires = (left, right)
    return param, wires

def rand_network(key, sizes):
    keys = random.split(key, len(sizes))
    dims = zip(keys, sizes[:-1], sizes[1:])
    return list(zip(*[rand_layer(*dim) for dim in dims]))

def predict(params, wires, inp, hard):
    active = inp
    for param, (left, right) in zip(params, wires):
        outs_l = jnp.dot(left, active)
        outs_r = jnp.dot(right, active)
        active = gate(outs_l, outs_r, param, hard)
    return active

predict_batch = vmap(predict, in_axes=(None, None, 0, None))

# l2 loss
def loss(params, wires, inp, out, hard):
    preds = predict_batch(params, wires, inp, hard)
    return jnp.mean(jnp.square(preds - out))

@functools.partial(jit, static_argnums=(4,))
def update_adamw(params, wires, x, y, opt, opt_state):
    # I think params has to be the first argument?
    grads = grad(loss)(params, wires, x, y, False)
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

def train_adamw(key, params, wires, epochs=3001, batch_size=512):
    import time
    keys = random.split(key, epochs)
    opt = optax.chain(
        optax.clip(100.0),
        optax.adamw(learning_rate=0.05, b1=0.9, b2=0.99, weight_decay=1e-2)
    )
    opt_state = opt.init(params)
    for (i, key_epoch) in enumerate(keys):
        key_train, key_accuracy = random.split(key_epoch)

        time_start = time.time()
        x = conway_sample_batch(key_train, batch_size)
        y = conway_kernel_batch(x)
        params, opt_state = update_adamw(params, wires, x, y, opt, opt_state)
        time_epoch = time.time() - time_start

        print(f"Epoch ({i+1}/{epochs}) in {time_epoch:.3g}s", end="   \r")
        if i % 1000 == 0: debug_loss(key_accuracy, params, wires, x, y)
        # if i % 10000 == 0: debug_params(params)
    return params

def debug_loss(key, params, wires, x, y):
    x_test = conway_sample_batch(key, x.shape[0])
    y_test = conway_kernel_batch(x_test)
    train_loss = loss(params, wires, x, y, False)
    test_loss = loss(params, wires, x_test, y_test, False)
    test_loss_hard = loss(params, wires, x_test, y_test, True)
    preds = predict_batch(params, wires, x_test, False)
    preds_hard = predict_batch(params, wires, x_test, True)
    print("[", *[f"{x:.3g}" for x in preds[0:5].flatten().tolist()], "]", preds_hard[0:5].flatten(), y_test[0:5].flatten())
    print(f"train_loss: {train_loss:.3g}", end="; ")
    print(f"test_loss: {test_loss:.3g}", end="; ")
    print(f"test_loss_hard: {test_loss_hard:.3g}")

def debug_params(params):
    for i, param in enumerate(params):
        print("LAYER", i, param.shape)
        for gate in param.T.tolist():
            for logic in gate:
                if logic > 1: print("█ ", end="")
                elif logic < 0: print("▁ ", end="")
                else: print("▄ ", end="")
            print()

def ext_gate_name(idx, l, r):
    names = [
        lambda a, b: "0",
        lambda a, b: f"{a} & {b}",
        lambda a, b: f"{a} & ~{b}",
        lambda a, b: a,
        lambda a, b: f"{b} & ~{a}",
        lambda a, b: b,
        lambda a, b: f"{a} ^ {b}",
        lambda a, b: f"{a} | {b}",
        lambda a, b: f"~({a} | {b})",
        lambda a, b: f"~({a} ^ {b})",
        lambda a, b: f"~{b}",
        lambda a, b: f"{a} | ~{b}",
        lambda a, b: f"~{a}",
        lambda a, b: f"{b} | ~{a}",
        lambda a, b: f"~({a} & {b})",
        lambda a, b: "~0",
    ]
    return names[idx](l, r)

def ext_add_deps(req, idx, l, r):
    deps = [
        lambda a, b: [],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a],
        lambda a, b: [a, b],
        lambda a, b: [b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [b],
        lambda a, b: [a, b],
        lambda a, b: [a],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [],
    ]
    for g in deps[idx](l, r):
        req.add(g)
    return req

def ext_gate(req, o, idx, l, r):
    name = ext_gate_name(idx, l, r)
    if o in req:
        req = ext_add_deps(req, idx, l, r)
        return req, f"cell {o} = {name};"
    return req, None

def ext_layer(req, param, left, right, layer):
    out = []
    for i, (g, l, r) in enumerate(zip(param.T, left, right)):
        idx_g = jnp.argmax(g, axis=0)
        idx_l = jnp.argmax(l, axis=0)
        idx_r = jnp.argmax(r, axis=0)
        req, instr = ext_gate(req, f"g_{layer+1}_{i}", idx_g, f"g_{layer}_{idx_l}", f"g_{layer}_{idx_r}")
        if instr is not None:
            out.append(instr)
    return req, out

def ext_logic(params, wires):
    out = []
    req = set(["g_24_0"])
    for layer, (param, (left, right)) in list(enumerate(zip(params, wires)))[::-1]:
        # print(param.shape, left.shape, right.shape)
        req, instrs = ext_layer(req, param, left, right, layer)
        out = instrs + out
    return out

if __name__ == "__main__":
    key = random.PRNGKey(379009)
    param_key, train_key = random.split(key)

    layer_sizes = [9, *([128] * 17), 64, 32, 16, 8, 4, 2, 1]
    params, wires = rand_network(param_key, layer_sizes)

    params_trained = train_adamw(train_key, params, wires)

    with open("gate.c", "w") as fout:
        for instr in ext_logic(params_trained, wires):
            fout.write(instr + "\n")
