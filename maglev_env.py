import os
from gymnasium import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

G = 5  # Gravitational constant
MAG_CONSTANT = 10  # Strength of the magnetic force
# TODO Clean up usage of dt so it's either passed in or a global const
DT = 0.05  # Time step for simulation


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
        self.observation_space = Box(-100, 100, shape=(9,))

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

    def step(self, new_charges_action):
        """
        Steps forward the environment using the new action

        Step return description - `tuple[state, reward, terminated, truncated, info]`
        Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
        """
        # For each electromagnet
        force = np.array([0., 0., 0.])
        for i, new_charge in enumerate(new_charges_action):
            # set new charge
            self.electromagnets[i].charge = new_charge
            # calc forces on ball from each pole
            force += self.ball.magnetic_force(self.electromagnets[i].Pole_N)
            force += self.ball.magnetic_force(self.electromagnets[i].Pole_S)

        force += G * self.ball.mass * np.array([0., 0., -1.])  # Gravity

        old_position = self.ball.position.copy()
        self.ball.update_positions_and_velocities(force)
        self.ball.adjust_if_collision(self.electromagnets)

        self.timesteps += 1

        reward = self.calculate_reward(old_position)
        terminated, truncated = self.check_done()
        obs = self.get_state()
        info = dict()

        return obs, reward, terminated, truncated, info

    def generate_random_point(self, x_range, y_range, z_range):
        # Generate random points within the specified range
        point = np.random.uniform(
            low=[x_range[0], y_range[0], z_range[0]],
            high=[x_range[1], y_range[1], z_range[1]],
            size=(3,))
        return point

    def get_state(self):
        """
        Returns a 9x1 np array of:
          (ball's XYZ position, ball's XYZ velocity, and desired XYZ position)
        """
        return np.concatenate((self.ball.position, self.ball.velocity, self.desired_position))

    def calculate_reward(self, pre_position):
        # penalize for distance from the desired position
        # reward for achieving setpoint and being steady
        prev_distance = np.linalg.norm(self.desired_position - pre_position)
        distance = np.linalg.norm(self.desired_position - self.ball.position)/np.linalg.norm((10,10,10))
        dist_reward = 1.0-distance
        reward = dist_reward**(self.timesteps)
        #print(dist_reward)

        # print(f"Total Reward: {reward}, Improve Reward: {improvement_reward}, Dist Reward: {dist_reward} ")
        return reward

    def render(self):
        self.draw_all()
        plt.pause(0.01)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an intial position of zeros for the 
        ball's position, velocity and desired position
        Returns:

        state: a numpy array of all zeros
        info: a variable containing any desired information to be passed on
        """
        if options is None:
            self.ball.position = np.array([0., 0., 0.])
            self.ball.velocity = np.array([0., 0., 0.])
            self.desired_position = np.array([0., 0., 0.])
        else:
            self.ball.position = self.generate_random_point(*options[0])
            self.desired_position = self.generate_random_point(*options[1])
            self.ball.velocity = np.array([0., 0., 0.])
        for electromag in self.electromagnets:
            electromag.charge = 1
        self.timesteps = 0
        # all values are zero
        state = np.concatenate((self.ball.position,self.ball.velocity,self.desired_position))
        info = dict()
        return state, info

    def check_done(self):
        """
        Checks if the environment has finished executing according to the following conditions:
            Elapsed time: If time runs out, truncated
            Bounding box: If the ball moves outside of a bounding box, truncated
            Success: If the ball reaches the final position with roughly zero velocity, terminate
        Returns:
            (Terminated (bool), Truncated (bool))
        """
        # Greater than 10 simulation seconds
        if self.timesteps * self.dt > 10:
            # print("time exceeded")
            return (False, True)
        # If the ball is further than 10 from 0,0,0
        if np.linalg.norm(self.ball.position) > 10:
            # print("out of bounds")
            return (False, True)
        # if the ball is within 0.01 distance of the desired position and has no velocity greater than 0.1
        # NOTE add condition where it has to be inside of the desired range for a certain amount of timesteps
        if (np.linalg.norm(self.ball.position - self.desired_position) < 0.01
            and np.linalg.norm(self.ball.velocity) < 0.1):
            # print("position reached successfully")
            return (True, False)
        return False, False

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
            size_scaling = abs(electromag.charge * (1/8 * 72 **2))
            # print("size of north pole: ", size_scaling)
            # print("size of south pole: ", (1/8 * 72 **2))
            self.ax.scatter(*electromag.Pole_N.position, c="blue", marker="o", s=size_scaling,alpha=0.75)
            # self.ax.scatter(*electromag.Pole_S.position, c="red", marker="o", s=size_scaling)
            # self.ax.scatter(*electromag.Pole_N.position, c="blue", marker="o", s=(1 / 8 * 72**2))
            self.ax.scatter(*electromag.Pole_S.position, c="red", marker="o", s=size_scaling,alpha=0.75)
        # draw position that we're trying to get the ball into
        self.ax.scatter(*self.desired_position, c="green", marker="o", s=100,alpha=0.5)
