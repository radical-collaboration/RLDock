import faulthandler
import sys
faulthandler.enable(file=sys.stderr, all_threads=True)

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

from rldock.environments.lactamase import LactamaseDocking
from config import config as envconf

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from rldock.voxel_policy.utils import kerasVoxelExtractor
tf = try_import_tf()

class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")


        layer_1 = kerasVoxelExtractor(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
import ray
from ray.rllib.agents.impala import impala
from ray.tune.logger import pretty_print


from ray.rllib.models import ModelCatalog
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--ncpu', type=int, default=4)
args = parser.parse_args()

ModelCatalog.register_custom_model("keras_model", MyKerasModel)


ray.init()
config = impala.DEFAULT_CONFIG.copy()
config['sample_batch_size'] = 62
config['train_batch_size'] = 496

config["num_data_loader_buffers"] =  4
config["minibatch_buffer_size"] =  4
config["num_sgd_iter"] = 4
config["replay_buffer_num_slots"] = 10
config["learner_queue_size"] = 32


config["num_gpus"] = args.ngpu # used for trainer process
config["num_workers"] = args.ncpu
config["num_cpus_per_worker"] = 1
config["num_gpus_per_worker"] = 0
config["num_cpus_for_driver"] = 1 # only used for tune.
config['num_envs_per_worker'] = 1
config["eager"] = False
config['env_config'] = envconf
# config['reuse_actors'] = True
config['model'] = {"custom_model": 'keras_model' }

trainer = impala.ImpalaTrainer(config=config, env=LactamaseDocking)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()

   if i % 3 == 0:
       print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)