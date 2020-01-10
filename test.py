import argparse
import os
import time

import numpy as np

from config import config
from rldock.environments.lactamase import LactamaseDocking

best_mean_reward, n_steps = -np.inf, 0


def make_dir(path):
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

    env = LactamaseDocking(config)

    iters = 1000
    start = time.time()
    obs = env.reset()
    name = 'ligand'
    fp_path = '/PycharmProjects/RLDock/'
    with open('run.pml', 'w') as fp:
        i = 0
        # with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
        #     cur_m  = env.cur_atom
        #     f.write(cur_m.toPDB())
        # fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
        # fp.write(", ligand" + name + ", " + str(i + 1) + "\n")
        i_adjust = 0
        for i in range(1, 100):
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            print(action, rewards, done)
            env.render(mode='human')

            atom = env.cur_atom

            if done:
                obs = env.reset()
                atom = env.cur_atom

            with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
                f.write(atom.toPDB())
            with open('pdbs_traj/test_p' + str(i) + '.pdb', 'w') as f:
                with open(env.receptor_refereence_file_name, 'r')as tfs:
                    f.writelines(tfs.readlines())
            fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
            fp.write(", ligand" + name + ", " + str(i + 1 - i_adjust) + "\n")
            fp.write("load " + fp_path + 'pdbs_traj/test_p' + str(i) + '.pdb ')
            fp.write(", protein" + name + ", " + str(i + 1 - i_adjust) + "\n")

end = time.time()
print("iters", iters / (end - start))

env.close()
