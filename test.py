from mpi4py import MPI
COMM =  MPI.COMM_WORLD

import argparse
import os

import numpy as np
from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
# from rldock.common.wrappers import DistributedPPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecFrameStack
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines.common.policies import register_policy

from config import config
from rldock.environments.lactamase import LactamaseDocking
from rldock.voxel_policy.actorcritic import CustomPolicy

best_mean_reward, n_steps = -np.inf, 0

def  make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('-e', type=int, default=10)
    return parser.parse_args()

from resnet import Resnet3DBuilder

if __name__ == '__main__':
    rn = Resnet3DBuilder.build_resnet_34((26,26,26,8), 400)
    # # Create log dir
    #
    # # # Create and wrap the environment
    # # args = getargs()
    # # # print(args)
    # # #
    # env = DummyVecEnv([lambda: LactamaseDocking(config)])
    # envs = SubprocVecEnv([lambda: LactamaseDocking(config)] * 8)
    #
    # model = PPO2(CustomPolicy, envs, verbose=2, tensorboard_log="tensorlogs/")
    # # model = DistributedPPO2(CustomPolicy, env, comm=COMM, verbose=2, tensorboard_log="tensorlogs/")
    # # model.learn(total_timesteps=3000)
    #
    # for i in range(10):
    #     model.learn(total_timesteps=1000)
    #
    #     obs = env.reset()
    #     for i in range(1, 200):
    #         action = model.predict(obs)
    #         obs, rewards, done, info = env.step(action)
    #         atom = env.render()
    #         if done:
    #             obs = env.reset()
    #
    #
    # env.close()
