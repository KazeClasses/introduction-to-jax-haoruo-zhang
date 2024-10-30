import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import matplotlib.pyplot as plt
import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree, Shaped, ArrayLike, Int
import equinox as eqx


class PendulumSimulation(eqx.Module):
    box_size: Float = 5.0
    ball_size: Float = 0.2
    image_size: Int = 64
    time: Float = 10.0
    dt: Float = 0.01
    save_interval: Int = 10

    @property
    def n_frames(self) -> Int:
        return int(self.time / (self.save_interval * self.dt))

    def __init__(
        self, box_size: Float = 5.0, ball_size: Float = 0.2, image_size: Int = 64
    ):
        self.box_size = box_size
        self.ball_size = ball_size
        self.image_size = image_size

    def ODE_system(
    self, t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        angle, angular_velocity = y
        gravity, length = args
        # Return a jnp.array instead of a list
        return jnp.array([angular_velocity, -gravity / length * jnp.sin(angle)])




    def simulate_pendulum(
        self,
        initial_angle: float,
        initial_velocity: float,
        gravity: float,
        length: float,
    ):
        # Define the ODE term
        term = dx.ODETerm(self.ODE_system)
        
        # Define the solver method
        solver = dx.Dopri5()
        
        # Initial conditions
        initial_state = jnp.array([initial_angle, initial_velocity])
        
        # Arguments for the ODE system
        args = (gravity, length)
        
        # Define when to save the trajectory data
        saveat = dx.SaveAt(t0=True, ts=jnp.arange(0, self.time, self.save_interval * self.dt))
        
        # Solve the differential equation
        sol = dx.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0,
            t1=self.time,
            y0=initial_state,
            args=args,
            dt0=self.dt,
            saveat=saveat,
        )
        
        return sol




    def render_pendulum(self,angle: float,angular_velocity: float,length: float,) -> jnp.ndarray:
        # Initialize an empty image
        image = jnp.zeros((self.image_size, self.image_size)).reshape(-1)

        # Generate a grid for pixel positions
        grid_x, grid_y = jnp.meshgrid(
            jnp.linspace(-self.box_size / 2, self.box_size / 2, self.image_size),
            jnp.linspace(-self.box_size / 2, self.box_size / 2, self.image_size),
        )

        # Stack grid coordinates to create (N, 2) array where N = image_size * image_size
        coordinates = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        # Compute the position of the pendulum bob based on angle and length
        position = jnp.array([length * jnp.sin(angle), -length * jnp.cos(angle)])

        # Calculate the distance from each pixel to the pendulum bob's position
        distance = jnp.linalg.norm(coordinates - position, axis=1)

        # Set pixels within the ball_size radius to 1.0 to represent the pendulum bob
        image = jnp.where(distance <= self.ball_size, 1.0, 0.0)

        # Reshape the flat array back into a 2D image
        return image.reshape(self.image_size, self.image_size)

    def generate_dataset(
        self,
        n_sims: Int,
        gravity: Float,
        length: Float,
    ) -> tuple[
        Float[Array, " n_samples 2 n_res n_res"],
        Float[Array, " n_samples 1 n_res n_res"],
    ]:
        inputs = []
        outputs = []
        for i in range(n_sims):
            # Generate random initial conditions
            initial_angle = jax.random.uniform(jax.random.PRNGKey(i), minval=-jnp.pi, maxval=jnp.pi)
            initial_velocity = jax.random.uniform(jax.random.PRNGKey(i + 1), minval=-1.0, maxval=1.0)
            
            solution = self.simulate_pendulum(initial_angle, initial_velocity, gravity, length)
            
            # Render frames
            frames = jnp.array([
                self.render_pendulum(sol_angle, sol_velocity, length)
                for sol_angle, sol_velocity in zip(solution.ys[:, 0], solution.ys[:, 1])
            ])
            
            inputs.append(jnp.stack([frames[:-2], frames[1:-1]], axis=1))
            outputs.append(frames[2:])
        
        return jnp.stack(inputs).reshape(
            -1, 2, self.image_size, self.image_size
        ).astype(jnp.float32), jnp.stack(outputs).reshape(-1, 1, self.image_size, self.image_size).astype(jnp.float32)


if __name__ == "__main__":
    pendulum = PendulumSimulation(image_size=64)
    sol = pendulum.simulate_pendulum(0.0, 0.0, 9.8, 1.0)
    image = pendulum.render_pendulum(sol.ys[0][0], sol.ys[1][0], 1.0)
    dataset = pendulum.generate_dataset(5, 9.8, 1.0)

