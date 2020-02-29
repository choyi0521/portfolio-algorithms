import gym
from ray.rllib.models.extra_spaces import Simplex

class StockTradingEnvironment(gym.Env):
    def __init__(self, n_items):
        super().__init__()
        self.action_space = Simplex(shape=(n_items,))
        # self.observation_space =

ste = StockTradingEnvironment(3)
print(ste.action_space.sample())
