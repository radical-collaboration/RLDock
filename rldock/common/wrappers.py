from stable_baselines.common.vec_env import VecEnv, DummyVecEnv
from rldock.common.utils import save_model_with_norm, load_model_with_norm
from stable_baselines.ppo2.ppo2 import PPO2, get_schedule_fn, safe_mean, Runner, swap_and_flatten
import multiprocessing
import numpy as np

import time
import sys
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

import mpi4py
from mpi4py import MPI


import copyreg

from mpi4py import MPI



class DistributedPPO2(PPO2):

    '''
    For now, pass this guy a local dummy vec env.
    '''
    def __init__(self, *args, **kwargs):
        self.comm = kwargs.pop("comm")
        super(DistributedPPO2, self).__init__(*args, **kwargs)


        self.env_runners = self.comm.Get_size() - 1
        self.rank = self.comm.Get_rank()

        self.n_envs = 100
        self.n_batch = 100


    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            # runner = DistributedRunner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)

            ctx = multiprocessing.get_context('spawn')
            q = ctx.Queue()
            p = ctx.Process(target=runDRunner, kwargs={ 'examples_queue': q}) #, 'env' : self.env, 'model' : self, 'n_steps' : self.n_steps, 'gamma' : self.gamma, 'lam':self.lam})
            p.start()
            print("STarted up queue from master")


            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            n_updates = total_timesteps // self.n_batch
            print("about to run...", n_updates, "updates and batch size", self.n_batch)
            for update in range(1, n_updates + 1):
                print("In loop.")
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount

                # pull from queue
                print("Pulling from quee...")
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = q.get(block=True)
                print("Got something!")

                self.num_timesteps += self.n_batch
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []

                #non-recurrent version
                update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                inds = np.arange(self.n_batch)
                for epoch_num in range(self.noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, self.n_batch, batch_size):
                        timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                        self.n_batch + start) // batch_size)
                        end = start + batch_size
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                             update=timestep, cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                ## BRODCAST WEIGHTS

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

def runDRunner(*args, **kwargs):
    DistributedRunner(*args, **kwargs)

class DistributedRunner:
    def __init__(self, *args, **kwargs):


        self.comm  = MPI.COMM_WORLD
        self.incoming_queue = kwargs.pop('examples_queue')
        # super(DistributedRunner, self).__init__(*args, **kwargs)

        self.recieve()

    def recieve(self):
        while(True):
            print("Waiting for message")
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            print("Got one!")
            self.incoming_queue.put(data)

    def broadcast_weights(self):
        pass

    def broadcast_model(self):
        pass


class MPISubprocVecEnv():
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multithread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """

    def __init__(self, env_fns, model, comm, args, n_steps, gamma, lam):
        self.comm = comm
        self.rank = comm.Get_rank()

        self.args = args

        env = DummyVecEnv(env_fns)
        self.model = model


        n_env = 1
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.batch_ob_shape = (n_env*n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = self.model.initial_state
        self.dones = [False for _ in range(n_env)]

        print("this is ", self.rank, "starting up.")
        self.run()


    def run(self):
        for i in range(self.n_steps):
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
            mb_states = self.states
            ep_infos = []
            for _ in range(self.n_steps):
                actions, values, _, neglogpacs = self.model.step(self.obs, self.states, self.dones)
                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.env.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
                self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
                for info in infos:
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)
                mb_rewards.append(rewards)
            # batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            last_values = self.model.value(self.obs, self.states, self.dones)
            # discount/bootstrap off value fn
            mb_advs = np.zeros_like(mb_rewards)
            true_reward = np.copy(mb_rewards)
            last_gae_lam = 0
            for step in reversed(range(self.n_steps)):
                if step == self.n_steps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[step + 1]
                    nextvalues = mb_values[step + 1]
                delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
                mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
            mb_returns = mb_advs + mb_values

            mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
                map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

            package = (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, None, ep_infos, true_reward)

            print("This is env", self.rank, "sending a message")
            self.comm.send(package, dest=0, tag=11)
            print("This is env", self.rank, "sent a message")

