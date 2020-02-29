import gym
from simplex import Simplex
from gym.space.box import Box
from gym.space.tuple import Tuple
import numpy as np


class StockTradingEnvironment(gym.Env):
    def __init__(self, lginc, stocks, features, obs_time_steps, obs_weight_num):
        super().__init__()
        self.lginc = lginc.copy()
        self.obs_time_steps = obs_time_steps
        self.obs_weight_num = obs_weight_num
        
        items = stocks + 1

        # action space
        self.action_space = Simplex(shape=(items, ))

        # observation space
        lginc_space = Box(
            low=np.full((items, features, obs_time_steps), -1),
            high=np.full((items, features, obs_time_steps), 1)
        )
        weight_space = Simplex(
            shape=(obs_weight_num, items)
        )
        self.observation_space = Tuple(lginc_space, weight_space)
    
    def _get_obs(self):
        return (self.lginc[:, :, self.t: self.t+self.obs_time_steps], self.weights)

    def reset(self):
        self.cumul_reward = 0
        self.t = 0
        self.weights = np.concatnate((np.ones((obs_weight_num, 1)), np.zeros((obs_weight_num, stocks))), axis=1)
        return self._get_obs()

    def step(self, action):
        reward = np.dot(action, self.lginc[self.t, :])
        self.cumul_reward += reward
        self.t += 1
        self.weights = np.concatnate((self.weights[:-1, :], [action]), axis=0)
        return self._get_obs(), reward, self.t == self.lginc.shape[-1] - self.obs_weight_num + 1



ste = StockTradingEnvironment(3)
print(ste.action_space.sample())
