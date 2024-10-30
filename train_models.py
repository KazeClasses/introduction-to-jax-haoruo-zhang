import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int, PRNGKeyArray
from generate_data import PendulumSimulation
import optax
from models import CNNEmulator, LatentODE

def loss_fn(model, batch):
    # Assuming batch is a tuple (input, target) where both are arrays of shape [batch_size, n_res, n_res]
    inputs, targets = batch
    predictions = model(inputs)
    loss = jnp.mean((predictions - targets) ** 2)  # MSE
    return loss

def train(
    model: CNNEmulator,
    dataset: Float[Array, " n_samples n_res n_res"],
    batch_size: Int,
    learning_rate: Float,
    num_epochs: Int,
    key: PRNGKeyArray,
) -> CNNEmulator:
    
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: CNNEmulator,
        opt_state: optax.OptState,
        batch: Float[Array, " n_samples n_res n_res"],
    ) -> tuple[CNNEmulator, optax.OptState, float]:
        # Compute loss and gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        
        # Apply updates to optimizer
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print("Training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(dataset) // batch_size
        for i in range(num_batches):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            model, opt_state, loss = make_step(model, opt_state, batch)
            epoch_loss += loss

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model

# Example usage
IMAGE_SIZE = 64
pendulum = PendulumSimulation(image_size=IMAGE_SIZE)
dataset = pendulum.generate_dataset(5, 9.8, 1.0)

CNNmodel = CNNEmulator(jax.random.PRNGKey(0))
trained_CNNmodel = train(CNNmodel, dataset, 4, 1e-3, 300, jax.random.PRNGKey(1))
CNNresult = trained_CNNmodel.rollout(dataset[0][0])
