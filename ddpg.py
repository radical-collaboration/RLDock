import argparse

from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack

from config import config
from rldock.common import utils
from rldock.environments.lactamase import LactamaseDocking
from rldock.voxel_policy.actorcritic import CustomPolicy


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('-l', type=str, default=None)
    parser.add_argument('-s', type=str, default='save_model')
    parser.add_argument('-e', type=int, default=10)
    parser.add_argument('-o', type=int, default=25)
    parser.add_argument('-r', type=float, default=2.5e-4)
    return parser.parse_args()


if __name__ == '__main__':

    # Create and wrap the environment
    args = getargs()
    print(args)

    env = VecFrameStack(VecNormalize(SubprocVecEnv([lambda: LactamaseDocking(config)] * args.p)), 4)
    if args.l is None:
        model = PPO2(CustomPolicy, env, verbose=2, tensorboard_log="tensorlogs/", learning_rate=args.r)
    else:
        model = utils.load_model_with_norm(PPO2, env, args.l)
    model.learn(total_timesteps=args.e)
    utils.save_model_with_norm(model, env, path=args.s)

    env.env_method('eval_ligands')
    env.env_method('disable_random')
    header = None
    cur_m = 0

    obs = env.reset()
    ligand_counter = 0
    lignad_name = 0
    fp_path = '/PycharmProjects/RLDock/'
    with open('run.pml', 'w') as fp:
        i = 0
        with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
            cur_m, name = env.env_method("render")[0]
            f.write(cur_m.toPDB())
        fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
        fp.write(", ligand" + name + ", " + str(i + 1) + "\n")
        i_adjust = 0
        for i in range(1, args.o):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            print(action, rewards, done)
            atom, name = env.env_method("render")[0]

            if done[0]:
                obs = env.reset()
                atom, name = env.env_method('render')[0]
                ligand_counter += 1
                i_adjust = 0

            with open('pdbs_traj/test' + str(i) + '.pdb', 'w') as f:
                f.write(atom.toPDB())
            fp.write("load " + fp_path + 'pdbs_traj/test' + str(i) + '.pdb ')
            fp.write(", ligand" + name + ", " + str(i + 1 - i_adjust) + "\n")

    env.close()
