import torch.nn as nn

import ray
from ray.rllib.agents import ddpg
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CustomTorchModel(nn.Module, TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        pass
    def forward(self, input_dict, state, seq_lens):
        pass
    def value_function(self):
        pass

ModelCatalog.register_custom_model("my_model", CustomTorchModel)

ray.init()
trainer = ddpg.DDPGTrainer(env="CartPole-v0", config={
    "use_pytorch": True,
    "model": {
        "custom_model": "my_model",
        "custom_options": {},  # extra options to pass to your model
    },
})
