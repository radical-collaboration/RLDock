from rldock.environments.lactamase import LactamaseDocking
import numpy as np
import os

import numpy as np
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
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
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


if  __name__ == '__main__':

    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment

    env = VecNormalize(SubprocVecEnv([lambda: Monitor(LactamaseDocking(), log_dir, allow_early_resets=True)]))

    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 64, 32])
    model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="tensorlogs/")
    model.learn(total_timesteps=1000,  callback=callback)
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

        for i in range(1,50):
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
                env.reset()


    env.close()


