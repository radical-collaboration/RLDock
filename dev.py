import os
import argparse
import os

import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

from rldock.environments.lactamase import LactamaseDocking
from rldock.voxel_policy.actorcritic import CustomPolicy

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('-s', type=str, default='save_model')
    parser.add_argument('-e', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':

    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    args = getargs()
    print(args)

    env = VecNormalize(DummyVecEnv([lambda: LactamaseDocking()]))
    model = PPO2(CustomPolicy, env, verbose=2, tensorboard_log="tensorlogs/")
    model.learn(total_timesteps=args.e, callback=callback)
    model.save(args.s)
    obs = env.reset()

    header = None
    states = []
    cur_m = 0

    env.reset()

    fp_path = '/Users/austin/PycharmProjects/RLDock/'
    with open('run.pml', 'w') as fp:
        i = 0
        with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
            cur_m = env.render()
            f.write(cur_m.toPDB())
        fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
        fp.write(", ligand, " + str(i + 1) + "\n")

        for i in range(1, 1000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            print(action, rewards, done)
            atom = env.render()
            header = atom.dump_header()
            states.append(atom.dump_coords())
            cur_m = atom

            with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
                f.write(cur_m.toPDB())
            fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
            fp.write(", ligand, " + str(i + 1) + "\n")

            if done:
                obs = env.reset()

    env.close()
