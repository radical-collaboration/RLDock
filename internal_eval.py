import faulthandler
import sys

faulthandler.enable(file=sys.stderr, all_threads=True)
import ray
from ray.rllib.agents.impala import impala
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo import appo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from argparse import ArgumentParser
from ray import tune
import copy
from config import config as envconf
from ray.tune.registry import register_env

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from rldock.voxel_policy.utils_tf2 import lrelu
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
from rldock.environments.lactamase import  LactamaseDocking
from resnet import Resnet3DBuilder
tf = try_import_tf()

class DeepDrug3D(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(DeepDrug3D, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)

        print(obs_space)
        self.inputs = [tf.keras.layers.Input(
            shape=(26, 27, 28, 8), name="observations"),


        tf.keras.layers.Input(shape=(8,), name='state_vec_obs')]
        h = tf.keras.layers.Conv3D(filters=32,  kernel_size=5, padding='valid', name='notconv1')(self.inputs[0])
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_2')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
            strides=None,
            padding='valid')(h)
        h = tf.keras.layers.Conv3D(filters=32,  kernel_size=5, padding='valid', name='notconv12')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
        h = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
            strides=None,
            padding='valid')(h)

        h = tf.keras.layers.Flatten()(h)

        layer_2 = tf.keras.layers.Dense(64, activation=lrelu)(self.inputs[1])
        layer_2 = tf.keras.layers.Concatenate()([layer_2, h])

        layer_2 = tf.keras.layers.Dense(64, activation=lrelu)(layer_2)
        layer_2 = tf.keras.layers.Dense(256, activation=lrelu)(layer_2)


        layer_4p = tf.keras.layers.Dense(256, activation='relu', name='ftp2')(layer_2)
        layer_5p = tf.keras.layers.Dense(64, activation=lrelu, name='ftp3')(layer_4p)

        layer_4v = tf.keras.layers.Dense(256, activation='relu', name='ftv2')(layer_2)
        layer_5v = tf.keras.layers.Dense(64, activation=lrelu, name='ftv3')(layer_4v)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation='sigmoid',
            kernel_initializer=normc_initializer(0.2))(layer_5p)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.1))(layer_5v)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):

        model_out, self._value_out = self.base_model([input_dict["obs"]['image'], input_dict['obs']['state_vector']])
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

        self.base_model.load_weights('my_model_weights.h5')

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# memory_story = 256.00  * 1e+9
# obj_store = 128.00 * 1e+9
# ray.init(memory=memory_story, object_store_memory=obj_store)
ray.init()

parser = ArgumentParser()
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--ncpu', type=int, default=4)
args = parser.parse_args()

ModelCatalog.register_custom_model("keras_model", MyKerasModel)
ModelCatalog.register_custom_model("deepdrug3d", DeepDrug3D)

def env_creator(env_config):
    return LactamaseDocking(env_config)  # return an env instance

register_env("lactamase_docking", env_creator)

config = ppo.DEFAULT_CONFIG.copy()
config['log_level'] = 'INFO'

ppo_conf = {"lambda": 0.95,
            "kl_coeff": 0,
            "sgd_minibatch_size": 96,
            "shuffle_sequences": True,
            "num_sgd_iter": 5,
            "lr": 1e-4,
            "vf_share_layers": True,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": None,
            "clip_param": 0.2,
            "kl_target": 0,
            "gamma" : 0.99
        }


config.update(ppo_conf)

config['sample_batch_size'] = 64
config['train_batch_size'] = 512

config["num_gpus"] = 0  # used for trainer process
config["num_workers"] = 1
config['num_envs_per_worker'] = 1
config['env_config'] = envconf
config['model'] = {"custom_model": 'deepdrug3d'}
config['horizon'] = envconf['max_steps']
from ray.rllib.agents.registry import get_agent_class
# trainer = impala.ImpalaTrainer(config=config, env='lactamase_docking')
#trainer = ppo.PPOTrainer(config=config, env="lactamase_docking")
checkpoint = '/PycharmProjects/RLDock/checkpoint_951/checkpoint-951'
#policy = trainer.get_policy()
#print(policy.model.base_model.summary())

config['env'] = 'lactamase_docking'
agent = get_agent_class('PPO')(env='lactamase_docking', config=config)
agent.restore(checkpoint)

env = LactamaseDocking(envconf)
# env.eval_ligands()

header = None

obs = env.reset()
ligand_counter = 0
lignad_name = 0
fp_path = '/PycharmProjects/RLDock/'

with open('run.pml', 'w') as fp:
    i = 0
    with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
        cur_m = copy.deepcopy(env.cur_atom)
        f.write(cur_m.toPDB())
    fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
    fp.write(", ligand" + str(ligand_counter )+ ", " + str(i + 1) + "\n")
    i_adjust = 0
    for i in range(1, 1000):
        action = agent.compute_action(obs)
        obs, rewards, done, info = env.step(action)

        print(action, rewards, done)
        atom = copy.deepcopy(env.cur_atom)

        if done:
            obs = env.reset()
            atom = copy.deepcopy(env.cur_atom)
            ligand_counter += 1
            i_adjust = 0

        with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
            f.write(atom.toPDB())
        fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
        fp.write(", ligand" + str(ligand_counter) + ", " + str(i + 1 - i_adjust) + "\n")

env.close()
