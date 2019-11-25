
import argparse
import os
import time

import numpy as np

from config import config
from rldock.environments.lactamase import LactamaseDocking

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
    # rn = Resnet3DBuilder.build_resnet_34((26,26,26,8), 400)
    # Create log dir

    # # Create and wrap the environment
    # args = getargs()
    # # print(args)
    # #
    env = LactamaseDocking(config)
    # envs = SubprocVecEnv([lambda: LactamaseDocking(config)] * 1)

    # model = PPO2(CustomPolicy, envs, verbose=2, tensorboard_log="tensorlogs/")
    # model = DistributedPPO2(CustomPolicy, env, comm=COMM, verbose=2, tensorboard_log="tensorlogs/")
    # model.learn(total_timesteps=3000)
    iters = 1000
    start = time.time()
    obs = env.reset()
    for i in range(iters):
        action = [env.action_space.sample()]
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')
        print(rewards)
        if done:
            obs = env.reset()
    end = time.time()
    print("iters", iters / (end - start))


    env.close()
