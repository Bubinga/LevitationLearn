import os
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import gymnasium as gym
from gymnasium import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def select_action_from_dist(mu_stdev_tup):
    """
    given a standard deviation and average for a
    probability distribution of the magnet's value's 

    expects:
        mu, stdev tuple of numbers for a magnet
    returns: 

    """
    mu, stdev = mu_stdev_tup
    charge_strength = torch.distributions.normal.Normal(mu, stdev).sample()
    return charge_strength

def resize_NN_output(distributions):
    """
    expects: 
        a 2N x 1 tensor of probabilitiy distributions for the action space output from
        the neural network
    returns:
        a 2 x N tensor of probability distributions where every cell is a pair 
        of a magnet's [mu, stdev]
    """
    electromag_ct = int(len(distributions.flatten().tolist()) / 2) # (length of flattened NN output / 2) = number of (mu, stdev) pairs 
    print("number of electromagnets: ", electromag_ct)
    out_tensor = distributions.view(electromag_ct, 2)
    print("output tensor after being resized: ", out_tensor)
    return out_tensor

   

def magnet_tudes(resized_output_tensor):
    """
    expects: 
        2xN tensor where one row of the tensor is a (mu, stdev) pair of numbers
        corresponding to the probability distribution for a given magnet's settings
    returns: 
        list of N numbers sampled from each magnet's probability distribution
    """
    magnet_settings = [] # NOTE this will need to be converted into a tensor
    
    
    for row in resized_output_tensor:
        
        row_in_normal_format = tuple(row.tolist())
        magnet_setting = select_action_from_dist(row_in_normal_format)
        magnet_settings.append(magnet_setting)
   
    return magnet_settings


class Policy:
    def select_action(self, state):
        pass # select your action here 

class StochasticPolicy(Policy):
    def update(self, states, actions, rewards):
        pass
    def get_probability(self, state, action):
        pass


class PolicyGradient(StochasticPolicy):
    def __init__(self, mdp, policy, alpha):
        super().__init__() # NOTE what is this inheriting from?
        self.alpha = alpha
        self.mdp = mdp
        self.policy = policy

    """ Want to generate and store entire episode trajectory to update the policy """
    def execute(self, episodes=10) -> None:
        for _ in range(1, episodes+1):
            actions = []
            states = [] 
            rewards = []

            state = env.reset()
            episode_reward = 0 

            done = False
            # score = 0
            while not done:
                action = env.action_space.sample()
                reward, done = env.step(action)
                # need to get a way to get the next state given an action


                # store this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                episode_reward += reward

            deltas = self.calculate_deltas(rewards)
            self.policy.update(states=states, actions=actions, deltas=deltas)

                # now you've gotten the output values, append them to a list to have memory of the episode
            print(f"Episode #{episode}: Score {score}")

    def calculate_deltas(self, rewards):
        """
        Generate a list of discounted future rewards at each step of an episode. discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma 
        We can populate discounted_rewards thsi way

        """ 
        T = len(rewards)
        discounted_future_rewards = [0 for _ in range(T)]
        # final discounted reward is reward obtained at that step
        discounted_future_rewards[T-1] = rewards[T-1]
        for t in reversed(range(0, T-1)):
            discounted_future_rewards[t] = (rewards[t] + discounted_future_rewards[t+1] * GAMMA) # NOTE finish this code


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

        # return self.velocity


class MagneticEnv(Env):
    def __init__(self, mag_coords, dt) -> None:
        super().__init__()
        # N = # of magnets, 1xN action space
        n = len(mag_coords)
        self.action_space = Box(-10, 10, shape=(n,)) # all possible electromagnet positions, n magnets long
        ### when samples from action space samples random  possible action then updates environ based on that new possible action occurring
        ### calcs force from each electromag from north and south poles, calcs gravity, then updates velocities. step function then returns loss and whether this ends or not.
        # XYZ current, XYZ velocities, and XYZ setpoint = 9x3
        self.observation_space = Box(-100, 100, shape=(3, 3)) # describe all possible states environ can eb in. XYZ set point, XYZ vels, XYZ current   
                                                            # NOTE should it be 9x3 instead?
        self.ball = MagneticTarget(1)
        self.electromagnets = []
        for magnet_pos in mag_coords:
            self.electromagnets.append(ElectroMagnet(1, magnet_pos))
        self.desired_position = np.array((0., 0., 0.))

        self.timesteps = 0
        self.dt = dt # NOTE inside of step we need to increment the time by one timestep (divide by) then 

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

        # increase the timestep by one every time we step forward  NOTE we also need a way to penalize the ball for taking too long to converge
        self.timesteps += 1 


        # NOTE is an info dictionary necessary? 

        
        return loss, done

    def render(self):
        self.draw_all()
        plt.pause(0.1)

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
        terminated = False
        finished = False
        
        if self.timesteps * self.dt > 10: # timesteps inside = 0, if still inside of this timestep increment the counter by one, otherwise reset the time 
            terminated = True
            finished = False
            return True
        # If the ball is further than 10 from 0,0,0
        if np.linalg.norm(self.ball.position) > 10:
            terminated = True 
            
            return True
        # if the ball is within 0.01 distance of the desired position and has no velocity greater than 0.1
        if (np.linalg.norm(self.ball.position - self.desired_position) < 0.01
            and np.linalg.norm(self.ball.velocity) < 0.1): # also add that it needs to be at this point for a certain amount of time. must maintain final position for some amount
            # of time b4 simulation is considered completed 
            return True
        return False
    
# NOTE a lot of dimension stuff is 3x3 apparently? 

    def draw_all(self):
        self.ax.clear()
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])
        self.ax.set_xlim([-5, 5])
        # draw target ball:
        self.ax.scatter(*self.ball.position, c="blue", marker="o")
        # draw electromagnets
        for electromag in self.electromagnets:
            self.ax.scatter(electromag.Pole_N.position, c="blue", marker="o", s=(1 / 8 * 72**2))
            self.ax.scatter(electromag.Pole_S.position, c="red", marker="o", s=(1 / 8 * 72**2))
        plt.show()


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


device = (
    "cuda"
    # if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    # else "cpu"
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
        action = env.action_space.sample()
        reward, done = env.step(action)
        score += reward
    print(f"Episode #{episode}: Score {score}")
