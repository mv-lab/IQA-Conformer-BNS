# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torchvision

# Gym
import gym
from gym.envs.classic_control import PendulumEnv
from torchvision import transforms

# CartPole Env
# classic state: [pos, cart speed, tip speed, angle]
# rgb State: 84 x 84
# Actions:
#   0: left
#   1: right
# Rewards:
#   step: +1
# The episode ends when the pole is more than 15 degrees from vertical, 
# or the cart moves more than 2.4 units from the center.
# CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
# End at step 200
class CartPole:

    def __init__(self, seed=None, img_size=(84, 84), mode="classic"):

        self.env = gym.make("CartPole-v0")
        self.actions = 2

        assert mode in ["classic", "rgb"]
        self.mode = mode

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Lambda(lambda x: x[:, 320:640, 280:-280]),
            transforms.Resize(img_size)
        ])

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()

        if self.mode == "classic":
            return torch.tensor(state)
        elif self.mode == "rgb":
            return self.transforms(self.env.render(mode="rgb_array").copy())
    
    def sample(self):

        return torch.tensor(self.env.action_space.sample())

    def step(self, action):

        # Forward Env
        state, reward, done, infos = self.env.step(action)

        if self.mode == "classic":
            return torch.tensor(state), reward, done
        elif self.mode == "rgb":
            return self.transforms(self.env.render(mode="rgb_array").copy()), reward, done

class Pendulum:

    """ Pendulum: https://www.gymlibrary.ml/pages/environments/classic_control/pendulum

    1 action:
    - Torque: -2:2

    3 observations:
    - x = cos(theta): -1:1
    - y = sin(angle): -1:1
    - Angular Velocity= -8:8

    Reward:
    r = -(theta**2 + 0.1 * theta_dt**2 + 0.001 * torque2)

    Start:
    The starting state is a random angle in [-pi, pi] and a random angular velocity in [-1,1].

    Ending
    The episode terminates at 200 time steps.
    
    """

    def __init__(self, seed=None, dt=0.05, img_size=(64, 64), mode="classic", max_steps=200, frames=3, state_reward=False):

        self.env = PendulumEnv()
        self.env.dt = dt
        self.actions = 1
        self.max_steps = max_steps
        self.frames = frames
        self.state_reward = state_reward
        self.num_actions = 1
        self.state_size = (3,)
        self.action_size = (1,)

        assert mode in ["classic", "rgb"]
        self.mode = mode

        if self.mode == "rgb":

            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                transforms.Resize(img_size)
            ])

            self.env.dt /= self.frames

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()
        self.steps = 0

        if self.mode == "classic":
            if self.state_reward:
                return [torch.tensor(state, dtype=torch.float32), torch.tensor([0], dtype=torch.float32)]
            else:
                return torch.tensor(state, dtype=torch.float32)
        elif self.mode == "rgb":
            if self.state_reward:
                return [self.transforms(self.env.render(mode="rgb_array").copy()).repeat(3, 1, 1), torch.tensor([0], dtype=torch.float32)]
            else:
                return self.transforms(self.env.render(mode="rgb_array").copy()).repeat(3, 1, 1)
    
    def sample(self):

        return self.env.action_space.sample()

    def step(self, action):

        self.steps += 1
        done = self.steps >= self.max_steps

        if self.mode == "classic":

            # Forward Env
            state, reward, _, _ = self.env.step([action])

            if self.state_reward:
                return [torch.tensor(state, dtype=torch.float32), torch.tensor([reward], dtype=torch.float32)], reward, done
            else:
                return torch.tensor(state, dtype=torch.float32), reward, done

        elif self.mode == "rgb":

            rgb_states = []
            rewards = 0.0

            for i in range(self.frames):

                state, reward, _, _ = self.env.step([action])
                rgb_states.append(self.transforms(self.env.render(mode="rgb_array").copy()))
                rewards += reward

            # Concat States
            rgb_states = torch.cat(rgb_states, dim=0)

            if self.state_reward:
                return [rgb_states, torch.tensor([rewards / self.frames], dtype=torch.float32)], rewards / self.frames, done
            else:
                return rgb_states, rewards / self.frames, done

###############################################################################
# Atari Environments
###############################################################################

# Breakout Env
# pip install 'stable-baselines3[extra]'
# pip install gym[atari,accept-rom-license]==0.19.0
# State: frames x 84 x 84
# Actions:
#   0: wait
#   1: start game / wait
#   2: right
#   3: left
# Rewards:
#   break block: + 1
#   other: + 0
# lives: 1 to 5
class Breakout:

    def __init__(self, frames=4, seed=None, img_size=(84, 84), lives=5):

        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.frames = frames
        self.actions = 4
        assert 1 <= lives <= 5
        self.lives = lives

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            transforms.Resize(img_size)
        ])

        if seed:
            self.env.seed(seed)

    def reset(self):

        self.env.reset()
        self.env_lives = 5

        # Start Game
        return self.step(1)[0]
    
    def sample(self):

        return torch.tensor(self.env.action_space.sample())

    def step(self, action):

        states = []
        rewards = 0.0
        done = False

        for _ in range(self.frames):

            state, reward, _, infos = self.env.step(action)
            states.append(self.transforms(state))

            # Lost Life
            if infos["ale.lives"] < self.env_lives:
                self.env_lives = infos["ale.lives"]
                reward = -1

                # Lost
                if self.env_lives == 5 - self.lives:
                    done = True

            rewards += reward

        # Concat States
        states = torch.cat(states, dim=0)

        return states, rewards, done

# Pong Env
# 2 players
# State: frames x 84 x 84
# Actions:
#   0 and 1: wait
#   2 and 4: up
#   3 and 5: down
# Rewards:
#   win game: +1
#   lose game: -1
# Ending: first player winning 21 games
class Pong:

    def __init__(self, frames=4, seed=None, img_size=(84, 84)):

        self.env = gym.make("PongNoFrameskip-v4")
        self.frames = frames
        self.actions = 6

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            transforms.Resize(img_size)
        ])

        if seed:
            self.env.seed(seed)

    def reset(self):

        self.env.reset()

        # Start Game
        return self.step(1)[0]
    
    def sample(self):

        return self.env.action_space.sample()

    def step(self, action):

        states = []
        rewards = 0.0
        done = False

        for _ in range(self.frames):
            state, reward, done_frame, infos = self.env.step(action)
            states.append(self.transforms(state))
            rewards += reward
            done = done or done_frame

        # Concat States
        states = torch.cat(states, dim=0)

        return states, rewards, done

###############################################################################
# Mujoco Environments
###############################################################################

class InvertedPendulum:

    """ Inverted Double Pendulum: https://www.gymlibrary.ml/pages/environments/mujoco/inverted_pendulum

    1 Action:
    - Force applied on the cart: -3:3

    4 observations
    - position of the cart along the linear surface
    - vertical angle of the pole on the cart
    - linear velocity of the cart
    - angular velocity of the pole on the cart

    rewards: 
    The goal is to make the inverted pendulum stand upright (within a certain angle limit) as long as possible - as such a reward of +1 is awarded for each timestep that the pole is upright.

    Termination:
    - The episode duration reaches 1000 timesteps.
    - Any of the state space values is no longer finite.
    - TThe absolutely value of the vertical angle between the pole and the cart is greater than 0.2 radian.

    """

    def __init__(self, seed=None):

        self.env = gym.make(
            id="InvertedPendulum-v2",
        )
        self.num_actions = 1
        self.state_size = (4,)
        self.action_size = (1,)

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()

        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):

        # Assert
        assert action.max() <= 3.0 and action.min() >= -3.0 

        # Forward Env
        state, reward, done, infos = self.env.step(action.tolist())

        return torch.tensor(state, dtype=torch.float32), reward, done

class InvertedDoublePendulum:

    """ Inverted Double Pendulum: https://www.gymlibrary.ml/pages/environments/mujoco/inverted_double_pendulum

    1 Action:
    - Force applied on the cart: -1:1

    11 observations

    3 rewards: The total reward returned is reward = alive_bonus - distance_penalty - velocity_penalty
    - alive_bonus: The goal is to make the second inverted pendulum stand upright (within a certain angle limit) as long as possible - as such a reward of +10 is awarded for each timestep that the second pole is upright.
    - distance_penalty: This reward is a measure of how far the tip of the second pendulum (the only free end) moves, and it is calculated as 0.01 * x2 + (y - 2)2, where x is the x-coordinate of the tip and y is the y-coordinate of the tip of the second pole.
    - velocity_penalty: A negative reward for penalising the agent if it moves too fast 0.001 * v12 + 0.005 * v2 2

    Termination:
    - The episode duration reaches 1000 timesteps.
    - Any of the state space values is no longer finite.
    - The y_coordinate of the tip of the second pole is less than or equal to 1. The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other).

    """

    def __init__(self, seed=None):

        self.env = gym.make(
            id="InvertedDoublePendulum-v2",
        )
        self.num_actions = 1
        self.state_size = (11,)
        self.action_size = (1,)

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()

        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):

        # Assert
        assert action.max() <= 1.0 and action.min() >= -1.0 

        # Forward Env
        state, reward, done, infos = self.env.step(action.tolist())

        return torch.tensor(state, dtype=torch.float32), reward / 10.0, done

# Swimmer
#
# Infos: https://www.gymlibrary.ml/pages/environments/mujoco/swimmer
#
# 2 Actions:
# - Torque applied on the first rotor: -1:1
# - Torque applied on the second rotor: -1:1
#
# 8 observations
# - angle of the front tip
# - angle of the second rotor
# - angle of the second rotor
# - velocity of the tip along the x-axis
# - velocity of the tip along the y-axis
# - angular velocity of front tip
# - angular velocity of second rotor
# - angular velocity of third rotor
#
# 2 rewards:
# - forward_reward: A reward of moving forward which is measured as forward_reward_weight * (x-coordinate before action - x-coordinate after action)/dt. 
# dt is the time between actions and is dependent on the frame_skip parameter (default is 4), where the frametime is 0.01 - making the default dt = 4 * 0.01 = 0.04. 
# This reward would be positive if the swimmer swims right as desired.
# - ctrl_cost:  A cost for penalising the swimmer if it takes actions that are too large. 
# It is measured as ctrl_cost_weight * sum(action2) where ctrl_cost_weight is a parameter set for the control and has a default value of 1e-4
#
# The total reward returned is reward = forward_reward - ctrl_cost and info will also contain the individual reward terms
#
# Termination: 
# - The episode duration reaches a 1000 timesteps
class Swimmer:

    def __init__(self, seed=None, never_ending=False):

        self.env = gym.make(
            id="Swimmer-v3",
            ctrl_cost_weight=0.005
        )
        self.num_actions = 2
        self.never_ending = never_ending
        self.state_size = (8,)
        self.action_size = (2,)

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()

        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):

        # Assert
        assert action.max() <= 1.0 and action.min() >= -1.0 

        # Forward Env
        state, reward, done, infos = self.env.step(action.tolist())

        return torch.tensor(state, dtype=torch.float32), reward, False if self.never_ending else done

# Half Cheetah
#
# Infos: https://www.gymlibrary.ml/pages/environments/mujoco/half_cheetah
#
# 6 Actions:
# - Torque applied on the back thigh rotor: -1:1
# - Torque applied on the back shin rotor: -1:1
# - Torque applied on the back foot rotor: -1:1
# - Torque applied on the front thigh rotor: -1:1
# - Torque applied on the front shin rotor: -1:1
# - Torque applied on the front foot rotor: -1:1
#
# 17 observations
#
# 2 rewards:
# - forward_reward: A reward of moving forward which is measured as forward_reward_weight * (x-coordinate before action - x-coordinate after action)/dt. 
# dt is the time between actions and is dependent on the frame_skip parameter (fixed to 5), where the frametime is 0.01 - making the default dt = 5 * 0.01 = 0.05. 
# This reward would be positive if the cheetah runs forward (right).
# - ctrl_cost: A cost for penalising the cheetah if it takes actions that are too large. 
# It is measured as ctrl_cost_weight * sum(action2) where ctrl_cost_weight is a parameter set for the control and has a default value of 0.1
#
# The total reward returned is reward = forward_reward - ctrl_cost and info will also contain the individual reward terms
#
# Termination: 
# - The episode duration reaches a 1000 timesteps
class HalfCheetah:

    def __init__(self, seed=None):

        self.env = gym.make(
            id="HalfCheetah-v3",
            #xml_file="/home/maximeburchi/Documents/git_clone/burchim/NeuralNets/half_cheetah.xml",
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.05
        )
        self.num_actions = 6
        self.state_size = (17,)
        self.action_size = (6,)

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()

        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):

        # Assert
        assert action.max() <= 1.0 and action.min() >= -1.0 

        # Forward Env
        state, reward, done, infos = self.env.step(action.tolist())

        return torch.tensor(state, dtype=torch.float32), reward, done

# Ant Env
#
# Infos: https://www.gymlibrary.ml/pages/environments/mujoco/ant
#
# 8 Actions:
# - Torque applied on the rotor between the torso and front left hip: -1:1
# - Torque applied on the rotor between the front left two links: -1:1
# - Torque applied on the rotor between the torso and front right hip: -1:1
# - Torque applied on the rotor between the front right two links: -1:1
# - Torque applied on the rotor between the torso and back left hip: -1:1
# - Torque applied on the rotor between the back left two links: -1:1
# - Torque applied on the rotor between the torso and back right hip: -1:1
# - Torque applied on the rotor between the back right two links: -1:1
#
# 111 Observations
#
# 4 rewards:
# - healthy_reward: 
# - forward_reward: (x_coord_before_action - x_coord_after_action) / dt, dt = 5 * 0.01 = 0.05, default frameskip = 5
# - ctrl_cost:  A negative reward for penalising the ant if it takes actions that are too large. It is measured as ctrl_cost_weight * sum(action ** 2) where ctr_cost_weight is a parameter set for the control and has a default value of 0.5.
# - contact_cost: A negative reward for penalising the ant if the external contact force is too large. It is calculated contact_cost_weight * sum(clip(external contact force to contact_force_range) ** 2).
#
# The total reward returned is reward = healthy_reward + forward_reward + ctrl_cost + contact_cost and info will also contain the individual reward terms.
#
# Termination: 
# - The episode duration reaches a 1000 timesteps
# - The ant is unhealthy
class Ant:

    def __init__(self, seed=None):

        self.env = gym.make("Ant-v3")
        self.num_actions = 8
        self.state_size = (111,)
        self.action_size = (8,)

        if seed:
            self.env.seed(seed)

    def reset(self):

        # Reset Env
        state = self.env.reset()

        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):

        # Forward Env
        state, reward, done, infos = self.env.step(action.tolist())

        #reward = infos["reward_forward"]

        #print()
        #print()
        #print(infos)
        #print()

        return torch.tensor(state, dtype=torch.float32), reward, done