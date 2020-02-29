class StockTradingEnvironment(gym.Env):
    def __init__(self, n_items):
        super().__init__()
        self.action_space = Simplex(shape=(n_items,))
        # self.observation_space =

ste = StockTradingEnvironment()
        