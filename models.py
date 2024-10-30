import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray


class CNNEmulator(eqx.Module):
    layers: list[eqx.Module]

    def __init__(self, key: jax.random.PRNGKey, hidden_dim: int = 4, n_res: int = 64):
        # Split key for initializing each layer independently
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Define layers: Convolution, Activation, and Output layers
        self.layers = [
            eqx.nn.Conv2d(in_channels=2, out_channels=hidden_dim, kernel_size=3, key=key1),
            eqx.nn.PReLU(),
            eqx.nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, key=key2),
            eqx.nn.Sigmoid()  # Adjust activation based on your output requirements
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Forward pass through each layer
        for layer in self.layers:
            x = layer(x)
        return x

