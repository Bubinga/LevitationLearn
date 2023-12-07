import os
from gymnasium import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# import pdb; pdb.set_trace()

G = 1  # Gravitational constant
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
        self.velocity = self.velocity + force / self.mass * DT
        self.position = self.position + self.velocity * DT

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

        # return self.velocity


class MagneticEnv(Env):
    def __init__(self, mag_coords, dt = DT) -> None:
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
        self.starting_position = np.array((0., 0., 0.))

        self.timesteps = 0
        self.success_time = 0
        self.dt = dt

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.tight_layout()


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

        terminated, truncated, in_goal = self.check_done()
        reward = self.calculate_reward(old_position, in_goal)
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
        return np.concatenate((self.ball.position, self.ball.velocity, self.desired_position),axis=0)

    def calculate_reward(self, pre_position, succeeded):
        def sign_square(x):
            """Squares a number while keeping the same sign"""
            return x * abs(x)
        # penalize for distance from the desired position
        # reward for achieving setpoint and being steady
        original_dist = np.linalg.norm(self.desired_position - self.starting_position)
        prev_distance = np.linalg.norm(self.desired_position - pre_position)
        distance = np.linalg.norm(self.desired_position - self.ball.position)
        success_reward = 1 if succeeded else 0
        dist_reward = (distance - original_dist)
        improvement_reward = (prev_distance-distance)/ DT
        # velocity_reward = -np.linalg.norm(self.ball.velocity) / 30
        
        reward = -dist_reward + 10*improvement_reward + 400 * success_reward

        # breakpoint()
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
        
        self.starting_position = self.ball.position.copy()
        self.timesteps = 0
        self.success_time = 0

        state = self.get_state()
        info = dict()
        return state, info

    def check_done(self):
        """
        Checks if the environment has finished executing according to the following conditions:
            Elapsed time: If time runs out, truncated
            Bounding box: If the ball moves outside of a bounding box, truncated
            Success: If the ball reaches the final position with roughly zero velocity, terminate
        Returns:
            (Terminated (bool), Truncated (bool), in_goal (bool))
        """
        # Greater than 10 simulation seconds
        in_goal = False
        terminated = False
        truncated = False
        
        if self.timesteps * self.dt > 10:
            truncated = True

        # If the ball is further than 5 from desired position
        if np.linalg.norm(self.ball.position - self.desired_position) > 5:
            truncated = True
        
        # if the ball is within 0.1 distance of the desired position and has no velocity greater than 0.1
        if np.linalg.norm(self.ball.position - self.desired_position) < 0.1:
            in_goal = True
            if self.success_time * self.dt > .5:
                terminated = True
            else:
                self.success_time += 1
        else:
            self.success_time = 0
            in_goal = False

        return (terminated, truncated, in_goal)

    def number_to_color(self, num):
        # Define the colormap (RdBu) and normalization
        cmap = plt.get_cmap('coolwarm')
        norm = Normalize(vmin=-5, vmax=5)
        
        # Map the number to a color
        color = cmap(norm(num))
        
        return color

    def draw_all(self):
        self.ax.clear()
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])
        self.ax.set_xlim([-5, 5])
        self.ax.axis(True)
        # draw target ball:
        self.ax.scatter(*self.ball.position, c="blue", marker="o")
        # draw electromagnets
        scale = 5
        for electromag in self.electromagnets:
            size_scaling = abs(1 * (1/8 * 72 **2))
            positions = zip(electromag.Pole_N.position, electromag.Pole_S.position)
            colors = [self.number_to_color(electromag.Pole_N.charge), self.number_to_color(electromag.Pole_S.charge)]
            self.ax.scatter(*positions, c=colors, marker="o", s=size_scaling,alpha=0.8)
        # draw position that we're trying to get the ball into
        self.ax.scatter(*self.desired_position, c="green", marker="o", s=100,alpha=0.5)
