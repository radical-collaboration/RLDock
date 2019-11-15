from mpi4py import MPI
import argparse

from rldock.common.wrappers import MPISubprocVecEnv, DistributedRunner, DistributedPPO2
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack, DummyVecEnv

from config import config
from rldock.common import utils
from rldock.environments.lactamase import LactamaseDocking
from rldock.voxel_policy.actorcritic import CustomPolicy


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('-l', type=str, default=None)
    parser.add_argument('-s', type=str, default='save_model')
    parser.add_argument('-e', type=int, default=1000)
    parser.add_argument('-o', type=int, default=25)
    parser.add_argument('-r', type=float, default=2.5e-4)
    return parser.parse_args()


if __name__ == '__main__':

    args = getargs()


    env = DummyVecEnv([lambda: LactamaseDocking(config)])
# if args.l is None:
    model = DistributedPPO2(CustomPolicy, env, verbose=2, n_steps=4, tensorboard_log="tensorlogs/", learning_rate=args.r, comm=comm)
    # else:
    #     model = utils.load_model_with_norm(PPO2, env, args.l)


    # utils.save_model_with_norm(model, env, path=args.s)

    print("Running now on to rank specalizing")

    if rank == 0:
        print("Learner sstarting")
        model.learn(total_timesteps=args.e)
    else:
        print("Subproc starting")
        MPISubprocVecEnv([lambda: LactamaseDocking(config)], model, comm, args, 100, model.gamma, model.lam)
