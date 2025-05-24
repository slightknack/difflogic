# From https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb#scrollTo=uWYAAFqsDV_Q

import jax
from jax import numpy as jnp
import flax.linen as nn
import functools
from collections import namedtuple
import optax
import einops

PASS_THROUGH_GATE = 3
DEFAULT_PASS_VALUE = 10.0
NUMBER_OF_GATES = 16

def get_moore_connections(key):
  """Generate Moore neighborhood connections for a 9x1 vector.

  Center element is at index 4 and connects to all other elements.
  """
  neighbors = jnp.array([0, 1, 2, 3, 5, 6, 7, 8])
  a = neighbors
  b = jnp.full_like(neighbors, 4)
  perm = jax.random.permutation(key, neighbors.shape[0])
  a = a[perm]
  b = b[perm]
  return a, b

# From https://github.com/Felix-Petersen/difflogic/tree/main/difflogic
def get_unique_connections(in_dim, out_dim, key):
    assert (
        out_dim * 2 >= in_dim
    )  # Number of neurons must not be smaller than half of inputs
    x = jnp.arange(in_dim)
    # Take pairs (0, 1), (2, 3), (4, 5), ...
    a = x[::2]
    b = x[1::2]
    m = min(a.shape[0], b.shape[0])
    a = a[:m]
    b = b[:m]
    # If needed, add pairs (1, 2), (3, 4), (5, 6), ...
    if a.shape[0] < out_dim:
        a_ = x[1::2]
        b_ = x[2::2]
        m = min(a_.shape[0], b_.shape[0])
        a = jnp.concatenate([a, a_[:m]])
        b = jnp.concatenate([b, b_[:m]])
    # If still needed, add pairs with larger offsets
    offset = 2
    while out_dim > a.shape[0] and offset < in_dim:
        a_ = x[:-offset]
        b_ = x[offset:]
        a = jnp.concatenate([a, a_])
        b = jnp.concatenate([b, b_])
        offset += 1

    if a.shape[0] >= out_dim:
        a = a[:out_dim]
        b = b[:out_dim]
    else:
        raise ValueError(
            f'Could not generate enough unique connections: {a.shape[0]} <'
            f' {out_dim}'
        )

    # Random permutation
    perm = jax.random.permutation(key, out_dim)
    a = a[perm]
    b = b[perm]

    return a, b

def bin_op_all_combinations(a, b):
    # Implementation of binary operations between two inputs for all the different operations
    return jnp.stack(
        [
            jnp.zeros_like(a),
            a * b,
            a - a * b,
            a,
            b - a * b,
            b,
            a + b - 2 * a * b,
            a + b - a * b,
            1 - (a + b - a * b),
            1 - (a + b - 2 * a * b),
            1 - b,
            1 - b + a * b,
            1 - a,
            1 - a + a * b,
            1 - a * b,
            jnp.ones_like(a),
        ],
        axis=-1,
    )

def bin_op_s(a, b, i_s):
    # Compute all possible operations
    combinations = bin_op_all_combinations(a, b)
    # Shape: (n_gate, n_possible_gates, 16)
    result = jax.numpy.sum(combinations * i_s[None, ...], axis=-1)
    return result

def decode_soft(weights):
  # From the weights vector compute the probability distribution of choosing each gate using softmax
  return nn.softmax(weights, axis=-1)

def decode_hard(weights):
    # Return the gate with maximum probability
    return jax.nn.one_hot(jnp.argmax(weights, axis=-1), 16)

def init_gates(
    n,
    num_gates=NUMBER_OF_GATES,
    pass_through_gate=PASS_THROUGH_GATE,
    default_pass_value=DEFAULT_PASS_VALUE,
):
    gates = jnp.zeros((n, num_gates))
    gates = gates.at[:, pass_through_gate].set(default_pass_value)
    return gates

def init_gate_layer(key, in_dim, out_dim, connections):
    # With 'random' connections the input of each gate are sampled randomly from the previous layer.
    if connections == 'random':
        key1, key2 = jax.random.split(key)
        c = jax.random.permutation(key2, 2 * out_dim) % in_dim
        c = jax.random.permutation(key1, in_dim)[c]
        c = c.reshape(2, out_dim)
        indices_a = c[0, :]
        indices_b = c[1, :]
    elif connections == 'unique':
        indices_a, indices_b = get_unique_connections(in_dim, out_dim, key)
    elif connections == 'first_kernel':
        indices_a, indices_b = get_moore_connections(key)
    else:
        raise ValueError(f'Connection type {connections} not implemented')

    wires = [indices_a, indices_b]
    gate_logits = init_gates(out_dim)
    return gate_logits, wires

def init_logic_gate_network(hyperparams, params, wires, key):
    for i, (in_dim, out_dim) in enumerate(
        zip(hyperparams['layers'][:-1], hyperparams['layers'][1:])
    ):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(
            subkey, int(in_dim), int(out_dim), hyperparams['connections'][i]
        )
        params.append(gate_logits)
        wires.append(gate_wires)

def init_perceive_network(hyperparams, params, wires, key):
    for i, (in_dim, out_dim) in enumerate(
        zip(hyperparams['layers'][:-1], hyperparams['layers'][1:])
    ):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(
            subkey, int(in_dim), int(out_dim), hyperparams['connections'][i]
        )
        """
        Replicate the gate logits for each of the 'n_kernels' perception kernels.
        This allows for parallel computation of the perception module,
        as all kernels share the same underlying structure and wiring.
        """
        params.append(
            gate_logits.repeat(hyperparams['n_kernels'], axis=0).reshape(
                hyperparams['n_kernels'], out_dim, NUMBER_OF_GATES
            )
        )
        wires.append(gate_wires)

def init_diff_logic_ca(hyperparams, key):
  key, subkey = jax.random.split(key)
  params = {'update': [], 'perceive': []}
  wires = {'update': [], 'perceive': []}
  init_logic_gate_network(
      hyperparams['update'], params['update'], wires['update'], subkey
  )
  key, subkey = jax.random.split(key)
  init_perceive_network(
      hyperparams['perceive'], params['perceive'], wires['perceive'], subkey
  )
  return params, wires

@functools.partial(jax.jit, static_argnums=(1,2))
def get_grid_patches(grid, patch_size, channel_dim, periodic):
    pad_size = (patch_size - 1) // 2
    padded_grid = jax.lax.cond(
        periodic,
        lambda g: jnp.pad(
            g, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="wrap"
        ),
        lambda g: jnp.pad(
            g,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="constant",
            constant_values=0,
        ),
        grid,
    )
    padded_grid = jnp.expand_dims(padded_grid, axis=0)
    patches = jax.lax.conv_general_dilated_patches(
        padded_grid,
        filter_shape=(patch_size, patch_size),
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )[0]

    # Rearrange to have (list, patch_size x patch_size, channel_dim)
    patches = einops.rearrange(patches, "x y (c f) -> (x y) f c", c=channel_dim)
    return patches

def run_layer(logits, wires, x, training):
    a = x[..., wires[0]]
    b = x[..., wires[1]]
    logits = jax.lax.cond(training, decode_soft, decode_hard, logits)
    out = bin_op_s(a, b, logits)
    return out

def run_update(params, wires, x, training):
    for g, c in zip(params, wires):
        x = run_layer(g, c, x, training)
    return x

def run_perceive(params, wires, x, training):
    run_layer_map = jax.vmap(run_layer, in_axes=(0, None, 0, None))
    x_prev = x
    x = x.T  # [channel_size, batch_size, patch_size]

    x = jnp.repeat(
        x[None, ...], params[0].shape[0], axis=0
    )  # [n_kernels, channel_size, batch_size, patch_size]

    for g, c in zip(params, wires):
        x = run_layer_map(g, c, x, training)

    x = einops.rearrange(
        x, 'k c s -> (c s k)'
    )  # [channel_size * patch_size * n_kernels]

    return jnp.concatenate(
        [x_prev[4, :], x], axis=-1
    )  # Concatenate the original input.

def run_circuit(params, wires, x, training):
    x = run_perceive(params['perceive'], wires['perceive'], x, training)
    x = run_update(params['update'], wires['update'], x, training)
    return x

# patches = [batch_size, n_patches, patch_size x patch_size, channels]
def v_run_circuit_patched(patches, params, wires, training):
    run_circuit_patch = jax.vmap(
        run_circuit, in_axes=(None, None, 0, None)
    )  # vmap over the patches
    x = run_circuit_patch(params, wires, patches, training)
    return x

@jax.jit
def run_async(grid, params, wires, training, periodic, key):
    patches = get_grid_patches(
        grid, patch_size=3, channel_dim=grid.shape[-1], periodic=periodic
    )
    x_new = v_run_circuit_patched(patches, params, wires, training)
    x_new = x_new.reshape(*grid.shape)
    update_mask_f32 = (
        jax.random.uniform(key, x_new[..., :1].shape) <= FIRE_RATE
    ).astype(jax.numpy.float32)
    x = grid * (1 - update_mask_f32) + x_new * update_mask_f32
    return x

@jax.jit
def run_sync(grid, params, wires, training, periodic):
    patches = get_grid_patches(
        grid, patch_size=3, channel_dim=grid.shape[-1], periodic=periodic
    )
    x_new = v_run_circuit_patched(patches, params, wires, training)
    x_new = x_new.reshape(*grid.shape)
    return x_new

@functools.partial(jax.jit, static_argnames=['num_steps', 'periodic', 'async_training'])
def run_iter_nca(grid, params, wires, training, periodic, num_steps, async_training, key):
    def body_fn(carry, i):
        grid, key = carry
        if async_training:
            key, subkey = jax.random.split(key)
            x = run_async(grid, params, wires, training, periodic, subkey)
        else:
            x = run_sync(grid, params, wires, training, periodic)
            return (x, key), 0

    (grid, key), _ = jax.lax.scan(
        body_fn, (grid, key), jnp.arange(0, num_steps, 1)
    )
    return grid

v_run_iter_nca = jax.vmap(
    run_iter_nca, in_axes=(0, None, None, None, None, None, None, None)
)

@jax.jit
def step(board):
    """Applies one step of Conway's Game of Life rules to the board.

    Args:
        board: A 2D array representing the game board (1 = live, 0 dead)

    Returns:
        board after one step of the game.

    """
    # Calculate the number of live neighbors for each cell.
    # jnp.roll shifts the board by the specified offsets (d)
    # to count neighbors in all 8 directions.
    n = sum(
        jnp.roll(board, d, (0, 1))
        for d in [
            (1, 0),  # Right
            (-1, 0),  # Left
            (0, 1),  # Down
            (0, -1),  # Up
            (1, 1),  # Down-right
            (-1, -1),  # Up-left
            (1, -1),  # Down-left
            (-1, 1),  # Up-right
        ]
    )

    # GOL rules:
    # - Birth: A dead cell with exactly 3 live neighbors becomes alive.
    # - Survive: A live cell with 2 or 3 live neighbors stays alive.
    # - Death: All other cells die.
    return (n == 3) | (board & (n == 2))  # Using bitwise OR and AND for efficiency

key = jax.random.PRNGKey(42)
board = jax.random.randint(
    key, shape=(64, 64), minval=0, maxval=2, dtype=jnp.uint8
)

hyperparams = {'perceive': {}, 'update': {}}
hyperparams['seed'] = 23
hyperparams['lr'] = 0.05
hyperparams['batch_size'] = 20
hyperparams['num_epochs'] = 3000
hyperparams['num_steps'] = 1
hyperparams['channels'] = 1
hyperparams['periodic'] = 1
hyperparams['perceive']['n_kernels'] = 16
hyperparams['perceive']['layers'] = [9, 8, 4, 2, 1]
hyperparams['perceive']['connections'] = [
    'first_kernel',
    'unique',
    'unique',
    'unique',
]
init = (
    hyperparams['perceive']['n_kernels']
    * hyperparams['channels']
    * hyperparams['perceive']['layers'][-1]
    + hyperparams['channels']
)
hyperparams['update']['layers'] = (
    [init] + [128] * 16 + [128, 64, 32, 16, 8, 4, 2, hyperparams['channels']]
)
hyperparams['update']['connections'] = ['unique'] * len(
    hyperparams['update']['layers']
)
hyperparams['async_training'] = False

key = jax.random.PRNGKey(hyperparams['seed'])
key, subkey = jax.random.split(key, 2)

@functools.partial(jax.jit, static_argnums=(1,))
def simulate_batch(boards, steps):
  def simulate_one(board):
    states = [board]
    for _ in range(steps):
      board = step(board)
      states.append(board)
    return jnp.stack(states)

  return jax.vmap(simulate_one)(boards)

def generate_binary_tensor():
  # Generate all possible combinations of 0s and 1s for 9 positions
  binary_numbers = jnp.arange(512)
  # Convert to binary and pad to 9 bits
  binary_array = (
      (binary_numbers[:, None] & (1 << jnp.arange(8, -1, -1))) > 0
  ).astype(jnp.float32)
  # Reshape to (512, 9, 1)
  tensor = binary_array.reshape(512, 9, 1)
  return tensor

initial_boards = generate_binary_tensor().reshape(-1, 3, 3).astype(jnp.uint8)

def sample_batch(key, trajectories, batch_size, state_size):
  n_samples = trajectories.shape[0]
  sample_idx = jax.random.randint(
      key, minval=0, maxval=n_samples, shape=(batch_size,)
  )
  init = jnp.zeros(
      (*trajectories[sample_idx, 0].shape, state_size), dtype=jnp.float32
  )
  init = init.at[..., 0].set(trajectories[sample_idx, 0])
  return init, trajectories[sample_idx, 1:, ..., None].astype(jnp.float32)


trajectories = simulate_batch(initial_boards, hyperparams['num_steps'])
batch_input, batch_output = sample_batch(
    subkey, trajectories, hyperparams['batch_size'], hyperparams['channels']
)

print(batch_input[8].reshape(3, 3))
print(batch_output[8, 0, :, :].reshape(3, 3))

# ii = get_grid_patches(batch_input[8], 3, 1, 1)
# print(ii[0].reshape(3, 3))

TrainState = namedtuple('TrainState', 'param opt_state key')

# Create optimizer
opt = optax.chain(
    optax.clip(100.0),  # Clips by value
    optax.adamw(
        learning_rate=hyperparams['lr'], b1=0.9, b2=0.99, weight_decay=1e-2
    ),
)

def init_state(hyperparams, opt, seed):
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    params, wires = init_diff_logic_ca(hyperparams, subkey)
    for p in params['update']:
        print(p.shape)
    opt_state = opt.init(params)
    return TrainState(params, opt_state, key), wires

def loss_f(params, wires, train_x, train_y, periodic, num_steps, async_training, key):
    def eval(params, training):
        y = v_run_iter_nca(
            train_x, params, wires, training, periodic, num_steps, async_training, key
        )
        return jax.numpy.square(y[..., 0] - train_y[..., 0]).sum()

    return eval(params, 1), {'hard': eval(params, 0)}

val_and_grad = jax.value_and_grad(loss_f, argnums=0, has_aux=True)

@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(
    train_state, train_x, train_y, wires, periodic, num_steps, async_training
):
    params, opt_state, key = train_state
    key, k1 = jax.random.split(key, 2)
    (loss, hard), dx = val_and_grad(
        params, wires, train_x, train_y, periodic, num_steps, async_training, k1
    )
    dx, opt_state = opt.update(dx, opt_state, params)
    new_params = optax.apply_updates(params, dx)
    return TrainState(new_params, opt_state, key), loss, hard

train_state, wires = init_state(hyperparams, opt, hyperparams['seed'])

loss_soft = []
loss_hard = []

for i in range(hyperparams['num_epochs']):
    key, sample_key = jax.random.split(key, 2)
    train_x, train_y = sample_batch(
        sample_key,
        trajectories,
        hyperparams['batch_size'],
        hyperparams['channels'],
    )
    train_state, soft_loss, hard_loss = train_step(
        train_state,
        train_x,
        train_y[:, 0, :, :],
        wires,
        hyperparams['periodic'],
        hyperparams['num_steps'],
        hyperparams['async_training'],
    )
    loss_soft.append(soft_loss)
    loss_hard.append(hard_loss['hard'])

    if i % 100 == 0:
        # clear_output(wait=True)
        # plot_training_progress(loss_soft, loss_hard, 1)
        print(i, soft_loss, hard_loss['hard'])
