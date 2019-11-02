import argparse

from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv

from config import config
from rldock.common import utils
from rldock.environments.lactamase import LactamaseDocking
from rldock.voxel_policy.actorcritic import CustomPolicy


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('-s', type=str, default='save_model')
    parser.add_argument('-e', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':

    # Create and wrap the environment
    args = getargs()
    print(args)

    env = VecNormalize(SubprocVecEnv([lambda: LactamaseDocking(config)] * args.p))
    model = PPO2(CustomPolicy, env, verbose=2, tensorboard_log="tensorlogs/")
    model.learn(total_timesteps=args.e)
    utils.save_model_with_norm(model, env, path=args.s)
    obs = env.reset()

    header = None
    states = []
    cur_m = 0

    env.reset()

    fp_path = '/Users/austin/PycharmProjects/RLDock/'
    with open('run.pml', 'w') as fp:
        i = 0
        with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
            cur_m = env.env_method("render")[0]
            f.write(cur_m.toPDB())
        fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
        fp.write(", ligand, " + str(i + 1) + "\n")

        for i in range(1, config['max_steps']):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            print(action, rewards, done)
            atom = env.env_method("render")[0]
            header = atom.dump_header()
            states.append(atom.dump_coords())
            cur_m = atom

            with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
                f.write(cur_m.toPDB())
            fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
            fp.write(", ligand, " + str(i + 1) + "\n")

            if done[0]:
                env.reset()

    env.close()
