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
    env = VecNormalize(DummyVecEnv([lambda: LactamaseDocking(config)] * 1))
    # model = DistributedPPO2(CustomPolicy, env, comm=COMM, verbose=2, tensorboard_log="tensorlogs/")
    # model.learn(total_timesteps=3000)

    
    obs = env.reset()

    fp_path = '/Users/austin/PycharmProjects/RLDock/'
    with open('run.pml', 'w') as fp:
        i = 0
        with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
            cur_m = env.env_method("render")[0]
            f.write(cur_m.toPDB())
        fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
        fp.write(", ligand, " + str(i + 1) + "\n")

        for i in range(1, 100):
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)

            print(action, rewards, done)
            atom = env.env_method("render")[0]
            header = atom.dump_header()
            # states.append(atom.dump_coords())
            cur_m = atom

            with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
                f.write(atom.toPDB())
            fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
            fp.write(", ligand, " + str(i + 1) + "\n")
            if done:
                obs = env.reset()
            print("hi")

    env.close()
