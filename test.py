from mpi4py import MPI
COMM =  MPI.COMM_WORLD

import argparse
import os

import numpy as np
# from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
# from rldock.common.wrappers import DistributedPPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecFrameStack
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines.common.policies import register_policy

from config import config
from rldock.environments.lactamase import LactamaseDocking
# from rldock.voxel_policy.actorcritic import CustomPolicy

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


if __name__ == '__main__':

    # Create log dir

    # # Create and wrap the environment
    # args = getargs()
    # # print(args)
    # #
    env = LactamaseDocking(config)
    # model = DistributedPPO2(CustomPolicy, env, comm=COMM, verbose=2, tensorboard_log="tensorlogs/")
    # model.learn(total_timesteps=3000)

    
    obs = env.reset()

    for i in range(1, 10000):
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        atom = env.render()
        if done:
            env.reset()


    env.close()
