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
from rldock.voxel_policy.utils import kerasVoxelExtractor, lrelu
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
        # ll = tf.keras.layers.BatchNormalization(name='bbn3')(layer_1)
        layer_3p = tf.keras.layers.Dense(128, activation='relu', name='ftp')(layer_1)
        layer_4p = tf.keras.layers.Dense(64, activation='relu', name='ftp2')(layer_3p)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_3v = tf.keras.layers.Dense(128, activation='relu', name='ftv')(layer_1)
        layer_4v = tf.keras.layers.Dense(64, activation='relu', name='ftv2')(layer_3v)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation='tanh',
            kernel_initializer=normc_initializer(0.1))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
import ray
from ray.rllib.agents.impala import impala
# from ray.rllib.agents.ppo import appo
from ray.tune.logger import pretty_print


from ray.rllib.models import ModelCatalog
from argparse import ArgumentParser

memory_story = 285.51  * 1e+9
obj_store = 140.63 * 1e+9
ray.init(memory=memory_story, object_store_memory=obj_store)


parser = ArgumentParser()
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--ncpu', type=int, default=4)
args = parser.parse_args()


from ray.tune.registry import register_env

ModelCatalog.register_custom_model("keras_model", MyKerasModel)
def env_creator(env_config):
    return LactamaseDocking(envconf)  # return an env instance

register_env("lactamase_docking", env_creator)

config = impala.DEFAULT_CONFIG.copy()
config["num_data_loader_buffers"] = 2
config["minibatch_buffer_size"] = 2
config["num_sgd_iter"] = 2
config["replay_proportion"] = 0.15
config["replay_buffer_num_slots"] = 512
config["learner_queue_size"] = 64
config["learner_queue_timeout"] =  300
config['log_level'] = 'DEBUG'

config['broadcast_interval'] = 5
config['max_sample_requests_in_flight_per_worker'] = 1
config['num_data_loader_buffers'] =  4
config['sample_batch_size'] = 320
config['train_batch_size']  = 1600
config["num_gpus"] = args.ngpu # used for trainer process
config["num_workers"] = args.ncpu
config['num_envs_per_worker'] = 6
config['env_config'] = envconf
config['model'] = {"custom_model": 'keras_model' }

trainer = impala.ImpalaTrainer(config=config, env='lactamase_docking')

policy = trainer.get_policy()
print(policy.model.base_model.summary())
# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()

   if i % 1 == 0:
       print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)