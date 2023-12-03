import os
import torch
from torch import nn
import gymnasium as gym
from gymnasium import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork(nn.Module):
    def __init__(self, input_count, mag_count):
        super().__init__()
        self.input_count = input_count
        self.mag_count = mag_count
        self.model = nn.Sequential(
            nn.Linear(input_count, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, mag_count),
        )

    def forward(self, x):
        return self.model(x)

    def loss_calc(self, path, desired_point):
        # calcualtes distance from each point in path to desired point
        total_loss = torch.tensor([0.0])
        for point in path:
            distance = torch.sqrt(torch.sum((point - desired_point) ** 2))
            total_loss += distance
        return total_loss

    def gen_training_data(self, data_count=100):
        # create start points with random coords in the range (-.05 - 0.05,-.05 - 0.05, 0 - 1)
        start_x = (np.rand(data_count) - 0.5) * 0.1
        start_y = (np.rand(data_count) - 0.5) * 0.1
        start_z = np.normal(mean=0.5, std=0.2, size=(1, data_count)).squeeze()
        start_z.clamp(0, 1)
        start = np.stack((start_x, start_y, start_z), dim=1)

        # create desired end points with random coords in the range (0,0, 0.1-0.9)
        finish_x, finish_y = np.zeros(data_count), np.zeros(data_count)
        finish_z = np.rand(data_count) * 0.8 + 0.1
        finish = np.stack((finish_x, finish_y, finish_z), dim=1)

        return np.stack((start, finish), dim=1)

class ElectroMagnet:
    def __init__(self, charge=1, position=np.array([0., 0., 0.])) -> None:
        self.position = position
        collision_radius = 1
        self.Pole_N = MagneticObject(collision_radius, charge, position + np.array((0, 0, 0.5)))
        self.Pole_S = MagneticObject(collision_radius, -charge, position + np.array((0, 0, -0.5)))
        self.charge = charge

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        self._charge = value
        self.Pole_N.charge = value
        self.Pole_S.charge = -value


class MagneticObject:
    def __init__(self, collision_radius, charge=1, position=np.array([0., 0., 0.]), mass=1) -> None:
        self.charge = charge
        self.position = position
        self.velocity = np.array([0., 0., 0.])
        self.mass = mass
        self.collision_radius = collision_radius

    def magnetic_force(self, other_mag):
        """Calculate force on self from other magnetic Object"""
        r = other_mag.position - self.position
        distance = np.linalg.norm(r)
        force = -MAG_CONSTANT* (self.charge * other_mag.charge)/ (distance**2) * r / distance
        return force


class MagneticTarget(MagneticObject):
    def __init__(self, collision_radius, charge=1, position=np.array((0, 0, 0)), mass=1) -> None:
        super().__init__(charge, collision_radius, position, mass)        

    # update it's own position, velocity, etc
    # Function to update positions and velocities
    def update_positions_and_velocities(self, force):
        self.position = self.position + self.velocity * DT
        self.velocity = self.velocity + force / self.mass * DT

    # Function to check for collision and perform a realistic 3D reflection
    def adjust_if_collision(self, other_mags: list[ElectroMagnet]):
        def is_colliding(other_mag):
            if (np.linalg.norm(other_mag.Pole_N.position - self.position) 
            < np.linalg.norm(other_mag.Pole_N.collision_radius - self.collision_radius)):
                return True
            elif (np.linalg.norm(other_mag.Pole_S.position - self.position) 
            < np.linalg.norm(other_mag.Pole_S.collision_radius - self.collision_radius)):
                return True
            return False
        
        for electromag in other_mags:
            if is_colliding(electromag):
                # TODO check sign is correct upon run
                n = self.position - electromag.position
                n /= np.linalg.norm(n)  # Normalize collision normal vector

                # Calculate relative velocity (electromag is stationary)
                v_rel = self.velocity - np.array((0., 0., 0.))

                # Calculate reflection using the formula: v' = −(2(n · v) n − v)
                DAMPING_COEFF = 0.5
                vel2_reflect = -(2 * np.dot(v_rel, n) * n - v_rel) * DAMPING_COEFF
                self.velocity = vel2_reflect

        # return self.velocity◘


class MagneticEnv(Env):
    def __init__(self, mag_coords, dt) -> None:
        super().__init__()
        # N = # of magnets, 1xN action space
        n = len(mag_coords)
        self.action_space = Box(-10, 10, shape=(n,))
        # XYZ current, XYZ velocities, and XYZ setpoint = 3 x 3
        self.observation_space = Box(-100, 100, shape=(3, 3))

        self.ball = MagneticTarget(1)
        self.electromagnets = []
        for magnet_pos in mag_coords:
            self.electromagnets.append(ElectroMagnet(1, magnet_pos))
        self.desired_position = np.array((0., 0., 0.))

        self.timesteps = 0
        self.dt = dt

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.tight_layout()
        # self.charges = (np.random.rand(n) - 0.5) * 2 #random charges between -1 and 1

    def step(self, new_charges):
        # For each electromagnet
        force = np.array([0., 0., 0.])
        for i, new_charge in enumerate(new_charges):
            # set new charge
            self.electromagnets[i].charge = new_charge
            # calc forces on ball from each pole
            force += self.ball.magnetic_force(self.electromagnets[i].Pole_N)
            force += self.ball.magnetic_force(self.electromagnets[i].Pole_S)

        force += G * self.ball.mass * np.array([0., 0., -1.])  # Gravity

        self.ball.update_positions_and_velocities(force)
        self.ball.adjust_if_collision(self.electromagnets)

        # distance loss
        loss = np.linalg.norm(self.desired_position - self.ball.position)**2
        done = self.check_done()

        return loss, done

    def render(self):
        self.draw_all()
        plt.pause(0.01)

    def reset(self):
        self.ball.position = np.array([0., 0., 0.])
        self.ball.velocity = np.array([0., 0., 0.])
        for electromag in self.electromagnets:
            electromag.charge = 1
        self.timesteps = 0

    def check_done(self):
        """
        Checks if the environment has finished executing according to the following conditions:
        Elapsed time: If time runs out, it is done
        Bounding box: If the ball moves outside of a bounding box, it is done
        Successful Run: If the ball reaches the final position with roughly zero velocity, it succeeded.
        """
        # Greater than 10 simulation seconds
        if self.timesteps / self.dt > 10:
            print("time exceeded")
            return True
        # If the ball is further than 10 from 0,0,0
        if np.linalg.norm(self.ball.position) > 10:
            print("out of bounds")
            return True
        # if the ball is within 0.01 distance of the desired position and has no velocity greater than 0.1
        if (np.linalg.norm(self.ball.position - self.desired_position) < 0.01
            and np.linalg.norm(self.ball.velocity) < 0.1):
            print("position reached successfully")
            return True
        return False

    def draw_all(self):
        self.ax.clear()
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])
        self.ax.set_xlim([-5, 5])
        self.ax.axis(True)
        # draw target ball:
        self.ax.scatter(*self.ball.position, c="blue", marker="o")
        # draw electromagnets
        for electromag in self.electromagnets:
            self.ax.scatter(*electromag.Pole_N.position, c="blue", marker="o", s=(1 / 8 * 72**2))
            self.ax.scatter(*electromag.Pole_S.position, c="red", marker="o", s=(1 / 8 * 72**2))


def end_condition(des_pos, ball_coord, time_spent):
    """
    expects:
    des_pos = (x,y,z) coords describing desired position
    ball_coord: (x,y,z) of ball_coordinates
    time_spent: int representing seconds since simulation began
    """
    pass

# Constants
G = 5  # Gravitational constant
MAG_CONSTANT = 10  # Strength of the magnetic force
DT = 0.05  # Time step for simulation

if __name__ == "main":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device") 

    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)

    mag_coords = [np.array([0.,0.,2.])]
    # TODO set the range of observation space to the range of the visualization
    env = MagneticEnv(mag_coords, DT)
    # model = NeuralNetwork(env.observation_space.shape[0],env.action_space.shape[0]).to(device)
    # training_data = model.gen_training_data(10)

    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            reward, done = env.step(action)
            score += reward
        print(f"Episode #{episode}: Score {score}")
