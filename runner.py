import argparse

import ray
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from config import config as envconf
from rldock.environments.lactamase import LactamaseDocking
from rnntest import MyKerasRNN

env = LactamaseDocking(envconf)

checkpoint = "/Users/austin/PPO_lactamase_docking_2019-12-09_15-09-10t3qmcnrc/checkpoint_1/checkpoint-1"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    return parser.parse_args()


def env_creator(env_config):
    print(env_config)
    return LactamaseDocking(env_config)
    # return an env instance


if __name__ == '__main__':
    register_env("lactamase_docking", env_creator)

    ray.init()
    # memory_story = 256.00  * 1e+9
    # obj_store = 128.00 * 1e+9
    # ray.init(memory=memory_story, object_store_memory=obj_store)
    args = get_args()
    ModelCatalog.register_custom_model("rnn", MyKerasRNN)

    d = {
        "env": 'lactamase_docking',
        'log_level': "INFO",
        "env_config": envconf,
        "gamma": 0.95,
        'eager': False,
        "num_gpus": 0,
        "train_batch_size": 64,
        "sample_batch_size": 64,
        'sgd_minibatch_size': 32,
        "num_workers": 2,
        "num_envs_per_worker": 1,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 16,
        "vf_loss_coeff": 5e-2,
        'vf_share_layers': True,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 20,
        }}
    ppo_config = ppo.DEFAULT_CONFIG
    ppo_config.update(d)


    get_dock_marks = []
    envconf['normalize'] = False
    workers = RolloutWorker(env_creator,  ppo.PPOTFPolicy,env_config=envconf, policy_config=d)
    # workers.restore(open(checkpoint, 'rb'))
    obs = env.reset()
    with open(args.o, 'w') as f:
        for i in workers.sample()['rewards']:
            f.write(str(i) + "\n")


    env.close()