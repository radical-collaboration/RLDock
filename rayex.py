import faulthandler
import sys

faulthandler.enable(file=sys.stderr, all_threads=True)
import ray
from ray.rllib.agents.impala import impala
from ray.rllib.agents.ppo import ppo
# from ray.rllib.agents.ppo import appo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ddpg import ddpg

from argparse import ArgumentParser
from ray import tune

from config import config as envconf
from ray.tune.registry import register_env

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from rldock.voxel_policy.utils_tf2 import lrelu
from ray.rllib.models.preprocessors import Preprocessor

from rldock.environments.lactamase import  LactamaseDocking
from resnet import Resnet3DBuilder
tf = try_import_tf()

class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):

        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_14 = Resnet3DBuilder.build_resnet_34(self.inputs, 400)
        self.model = tf.keras.models.Model(self.inputs, layer_14)

        return (400)

    def transform(self, observation):
        return self.model(observation)

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)


class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        layer_14 =  Resnet3DBuilder.build_resnet_34(self.inputs, 400)


        layer_2 = tf.keras.layers.Flatten()(layer_14)
        layer_3p = tf.keras.layers.Dense(256, activation='relu', name='ftp')(layer_2)
        layer_4p = tf.keras.layers.Dense(128, activation='relu', name='ftp2')(layer_3p)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_3v = tf.keras.layers.Dense(256, activation='relu', name='ftv')(layer_2)
        layer_4v = tf.keras.layers.Dense(128, activation='relu', name='ftv2')(layer_3v)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation='hard_sigmoid',
            kernel_initializer=normc_initializer(0.1))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.base_model.load_weights('my_model_weights.h5')

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        layer_14 =  Resnet3DBuilder.build_resnet_34(self.inputs, 400)


        layer_2 = tf.keras.layers.Flatten()(layer_14)
        layer_3p = tf.keras.layers.Dense(256, activation='relu', name='ftp')(layer_2)
        layer_4p = tf.keras.layers.Dense(128, activation='relu', name='ftp2')(layer_3p)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_3v = tf.keras.layers.Dense(256, activation='relu', name='ftv')(layer_2)
        layer_4v = tf.keras.layers.Dense(128, activation='relu', name='ftv2')(layer_3v)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation='hard_sigmoid',
            kernel_initializer=normc_initializer(0.1))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])


        #
        # d = torch.load("models/resnet-34-kinetics-cpu.pth")['state_dict']
        # convs = (list(filter( lambda x : "conv" in x, d.keys())))
        #
        # conv1_count = 1
        # for layer in model.layers:
        #     if isinstance(layer, tf.keras.layers.Conv3D):
        #         try:
        #             py_weight = d[convs.pop(0)].permute((4, 3, 2, 1, 0)).numpy()
        #             if conv1_count == 1:
        #                 print("editing weight will take a bit....")
        #                 py_weight = torch.from_numpy(py_weight).repeat((1,1,1,3,1)).numpy()[:7, :7, :7, :8, :64]
        #
        #             # py_bias = d['conv' + str(conv1_count) + ".weight"].numpy()
        #             print("torch", py_weight.shape)
        #
        #             print(layer.get_weights()[0].shape, layer.get_weights()[1].shape)
        #             conv1_count += 1
        #             try:
        #                 layer.set_weights([py_weight, layer.get_weights()[1]])
        #             except ValueError as e:
        #                 print(e)
        #         except IndexError:
        #             print("no torch", print(layer, layer.get_weights()[0].shape))

        self.base_model.load_weights('my_model_weights.h5')

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# memory_story = 200.00  * 1e+9
# obj_store = 64.00 * 1e+9
# ray.init(memory=memory_story, object_store_memory=obj_store)
ray.init()

parser = ArgumentParser()
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--ncpu', type=int, default=4)
args = parser.parse_args()

ModelCatalog.register_custom_model("keras_model", MyKerasModel)

def env_creator(env_config):
    return LactamaseDocking(env_config)  # return an env instance
register_env("lactamase_docking", env_creator)

config = ddpg.DEFAULT_CONFIG.copy()
config['log_level'] = 'DEBUG'

# ppo_conf = {"lambda": 0.95,
#     "kl_coeff": 0.2,
#     "sgd_minibatch_size": 512,
#     "shuffle_sequences": True,
#     "num_sgd_iter": 10,
#     "lr": 1e-4,
#     "lr_schedule": None,
#     "vf_share_layers": False,
#     "vf_loss_coeff": 0.5,
#     "entropy_coeff": 0.01,
#     "entropy_coeff_schedule": None,
#     "clip_param": 0.3,
#     "vf_clip_param": 5.0,
#     "grad_clip": 10.0,
#     "kl_target": 0.01}
#
# config.update(ppo_conf)

# config['sample_batch_size'] = 64
# config['train_batch_size'] = 1024

config["num_gpus"] = args.ngpu  # used for trainer process
config["num_workers"] = args.ncpu
config['num_envs_per_worker'] = 4
config['use_state_preprocessor'] = True

config['env_config'] = envconf
config['model'] = {"custom_model": 'my_prep'}
config['horizon'] = envconf['max_steps'] + 2

trainer = ddpg.DDPGTrainer(config=config, env='lactamase_docking')
# trainer = ppo.PPOTrainer(config=config, env="lactamase_docking")
# trainer.restore('/homes/aclyde11/ray_results/PPO_lactamase_docking_2019-11-18_13-40-14ihwtk2lw/checkpoint_51/checkpoint-51')
policy = trainer.get_policy()
print(policy.model.base_model.summary())

for i in range(1000):
    result = trainer.train()

    if i % 1 == 0:
        print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
