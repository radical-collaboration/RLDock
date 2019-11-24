import copy

import gym
import numpy as np
from gym import spaces
import random
from random import randint
from rldock.environments.LPDB import LigandPDB
from rldock.environments.utils import MultiScorer, Voxelizer, l2_action, MinMax
import glob
import math

# using 6DPT pdb from Lyu et al. (2019, nature)
class LactamaseDocking(gym.Env):
    metadata = {'render.modes': ['human']}

    ## Init the object
    def __init__(self, config ):
        super(LactamaseDocking, self).__init__()
        self.config = config
        self.viewer = None

        dims = np.array(config['bp_dimension']).flatten().astype(np.float32)
        self.random_space_init = spaces.Box(low=-0.5 * dims,
                                            high=0.5 * dims,
                                            dtype=np.float32)

        #rotations from 0 to 2pi
        self.random_space_rot = spaces.Box(low=0,
                                           high=2 * 3.1415926,
                                           dtype=np.float32,
                                           shape=(3,1))

        lows = -1 * np.array([1] * 9, dtype=np.float32)
        highs = np.array([1] * 9, dtype=np.float32)



        self.use_random = True

        if config['discrete']:
            self.actions_multiplier = np.array([config['action_space_d'][i] / (config['K_trans'] - 1) for i in range(3)]
                                               + [config['action_space_r'][i] for i in range(6)], dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([config['K_trans']] * 3 + [config['K_theta']] * 6)


        else:
            self.action_space = spaces.Box(low=lows,
                                           high=highs,
                                           dtype=np.float32)


        self.observation_space = spaces.Dict({"image" : spaces.Box(low=0, high=2, shape=config['output_size'], #shape=(29, 24, 27, 16),
                                                dtype=np.float32),
                                              "state_vector" : spaces.Box(low=np.array([-31, 0], dtype=np.float32),
                                                                          high=np.array([31, 1.1], dtype=np.float32))
                                            }
                                        )

        self.voxelizer = Voxelizer(config['protein_wo_ligand'], config)
        self.oe_scorer = MultiScorer(config['oe_box']) # takes input as pdb string of just ligand
        self.minmaxs = [MinMax(-278, -8.45), MinMax(-1.3, 306.15), MinMax(-17.52, 161.49), MinMax(-2, 25.3)]


        self.reference_ligand = LigandPDB.parse(config['ligand'])
        self.reference_centers = self.reference_ligand.get_center()
        self.atom_center =  LigandPDB.parse(config['ligand'])
        self.names = []

        if config['random_ligand_folder'] is not None:
            self.train_ligands()
        else:
            self.rligands = None

        self.cur_atom = copy.deepcopy(self.atom_center)
        self.trans = [0,0,0]
        self.rot   = [0,0,0]
        self.rot   = [0,0,0]
        self.steps = 0
        self.last_score = 0
        self.cur_reward_sum = 0
        self.name = ""

    def reset_ligand(self, newlig):
        """
        :param newlig: take a LPBD ligand and transform it to center reference
        :return: new LPDB
        """
        x,y,z  = newlig.get_center()
        return newlig.translate(self.reference_centers[0] - x , self.reference_centers[1] - y, self.reference_centers[2] - z)

    def align_rot(self):
        """
            Aligns interal rotations given angluar def in current version
        """
        for i in range(3):
            if self.rot[i] < 0:
                self.rot[i] = 2*3.14159265 + self.rot[i]
            self.rot[i] = self.rot[i] % (2 * 3.14159265)

    def get_action(self, action):
        """
        Override this function if you want to modify the action in some deterministic sense...
        :param action: action from step funtion
        :return: action
        """
        if self.config['discrete']:
            action = np.array(action) * self.actions_multiplier
        action = np.array(action).flatten()
        return action


    def get_penalty_from_overlap(self, obs):
        """
        Evaluates from obs to create a reward value. Do not scale this value, save for weighting later in step function.
        :param obs: obs from model, or hidden env state
        :return: penalty for overlap, positive value
        """
        if np.max(obs[:,:,:,-1]) == 2:
            return 1.0
        return 0.0

    def oe_score_combine(self, oescores, average=False):
        r = 0
        for i in range(len(oescores)):
            self.minmaxs[i].update(oescores[i])
            mins,maxs = self.minmaxs[i]()
            if oescores[i] > self.minmaxs[i].eps:
                norm_score = 0
            else:
                norm_score = (oescores[i] - maxs) / (maxs - mins)
            r += norm_score

        if average:
            r = r / len(oescores)
        return r

    @staticmethod
    def Nq(q):
        t = np.linalg.norm(q)
        if t == 0:
            return np.zeros(q.shape)
        return q / t

    @staticmethod
    def isRotationMatrix(M, eps = 1e-2):
        tag = False
        I = np.identity(M.shape[0])
        if np.all( np.abs(np.matmul(M, M.T) - I) <= eps) and (np.abs(np.linalg.det(M) - 1) <= eps): tag = True
        else:
            print('fail', M, np.abs(np.matmul(M, M.T) - I), np.abs(np.linalg.det(M) - 1))
        return tag

        #https: // arxiv.org / pdf / 1812.07035.pdf
    def get_rotation(self, rot):
        a_1 = np.array(rot[:3], dtype=np.float64)
        a_2 = np.array(rot[3:], dtype=np.float64)

        b_1 = self.Nq(a_1)
        b_2 = self.Nq( a_2 - np.dot(b_1, a_2) * b_1)
        b_3 = np.cross(b_1, b_2)

        M = np.stack([b_1, b_2, b_3]).T

        if self.isRotationMatrix(M):
            return M.astype(np.float32)
        else:
            print("Error. :(")
            return np.identity(M.shape[0], dtype=np.float32)

    def step(self, action):
        if np.any(np.isnan(action)):
            print(action)
            print("ERROR, nan action from get action")
            exit()

        action = self.get_action(action)
        assert(action.shape[0] == 9)


        self.trans[0] += action[0]
        self.trans[1] += action[1]
        self.trans[2] += action[2]
        self.rot = self.get_rotation(action[3:])

        self.cur_atom = self.cur_atom.translate(action[0], action[1], action[2])
        self.cur_atom = self.cur_atom.rotateM(self.rot)
        self.steps += 1

        oe_score = self.oe_scorer(self.cur_atom.toPDB())
        oe_score =  self.oe_score_combine(oe_score)
        reset = self.decide_reset(oe_score)

        self.last_score = oe_score
        obs = self.get_obs()

        w1 = float(1.0)
        w2 = 0
        w3 = 0

        reward = w1 * (-1.0 * oe_score) - w2 * l2_action(action) - w3 * self.get_penalty_from_overlap(obs)

        self.last_reward = reward
        self.cur_reward_sum += reward

        obs = {'image' : obs, 'state_vector' : self.get_state_vector()}

        return obs,\
               reward,\
               reset, \
               {}

    def decide_reset(self, score):
         return (self.steps > self.config['max_steps']) or (not self.check_atom_in_box())

    def get_reward_from_ChemGauss4(self, score, reset=False):
        # boost = 5 if self.steps > self.config['max_steps'] - 3 else 1
        score = -1 * score
        if score < -25:
            return 0
        elif score < 0:
            return 0.01
        else:
            return float(score) * 1



    def get_state_vector(self):
        max_steps = self.steps / self.config['max_steps']
        return np.array([float(np.clip(self.last_score, -30, 30)), max_steps]).astype(np.float32)


    def reset(self, random=0, many_ligands = True):
        if many_ligands and self.rligands != None and self.use_random:
            idz = randint(0, len(self.rligands) - 1)
            start_atom = copy.deepcopy(self.rligands[idz])
            self.name = self.names[idz]

        elif many_ligands and self.rligands != None :
            start_atom = copy.deepcopy(self.rligands.pop(0))
            self.name = self.names.pop(0)
        else:
            start_atom = copy.deepcopy(self.atom_center)

        if random is not None and float(random) != 0:
            x,y,z, = self.random_space_init.sample().flatten().ravel() * float(random)
            x_theta, y_theta, z_theta = self.random_space_rot.sample().flatten().ravel() * float(random)
            self.trans = [x,y,z]
            self.trans_ = [x, y, z]
            self.rot_ = [x_theta, y_theta, z_theta]
            random_pos = start_atom.translate(x,y,z)
            random_pos = random_pos.rotate(theta_x=x_theta, theta_y=y_theta, theta_z=z_theta)
        else:
            self.trans = [0,0,0]
            self.trans_ = [0,0,0]
            self.rot_ = [0, 0, 0]
            random_pos = start_atom

        self.trans = [0, 0, 0]
        self.cur_atom = random_pos
        self.last_score = self.oe_score_combine(self.oe_scorer(self.cur_atom.toPDB()))
        self.steps = 0
        self.cur_reward_sum=0
        self.last_reward = 0
        self.next_exit = False
        self.decay_v = 1.0
        return {'image' : self.get_obs(), 'state_vector' : self.get_state_vector()}

    def get_obs(self, quantity='all'):
        x= self.voxelizer(self.cur_atom.toPDB(), quantity=quantity).squeeze(0).astype(np.float32)
        if self.config['debug']:
            print("SHAPE", x.shape)
        return x

    def render(self, mode='human'):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        obs = (self.get_obs(quantity='ligand')[:,:,:,-1]).squeeze()
        obs1 = (self.get_obs(quantity='protein')[:,:,:,-1]).squeeze()
        # np.save("/Users/austin/obs.npy", obs)
        # np.save("/Users/austin/pro.npy", obs1)
        print(obs.shape)
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.gca(projection='3d')
        canvas = FigureCanvas(fig)

        coords_x = []
        coords_y = []
        coords_z = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for z in range(obs.shape[2]):
                    if obs[i,j,z] == 1:
                        coords_x.append(i)
                        coords_y.append(j)
                        coords_z.append(z)

        coords_x1 = []
        coords_y1 = []
        coords_z1 = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for z in range(obs.shape[2]):
                    if obs1[i,j,z] == 1:
                        coords_x1.append(i)
                        coords_y1.append(j)
                        coords_z1.append(z)

        ax.set_title("Current step:" + str(self.steps) + ", Curr Reward" + str(self.last_reward) + ', Curr RSUm' + str(self.cur_reward_sum)+ 'score'+ str(self.last_score))
        try:
            ax.plot_trisurf(coords_x, coords_y, coords_z, linewidth=0.2, antialiased=True)
            ax.plot_trisurf(coords_x1, coords_y1, coords_z1, linewidth=0.2, antialiased=True, alpha=0.5)

        except:
            pass

        ax.set_xlim(0,25)
        ax.set_ylim(0,26)
        ax.set_zlim(0,27)
        # fig.show()
        canvas.draw()  # draw the canvas, cache the renderer
        width , height = fig.get_size_inches() * fig.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(100 * 10, 100 * 10, 3)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        pass

    def check_atom_in_box(self):
        return self.random_space_init.contains(self.trans_)

    def disable_random(self):
        self.use_random = False

    def eval_ligands(self):
        self.rligands = glob.glob(self.config['random_ligand_folder_test'] + "/*.pdb")
        self.names = copy.deepcopy(self.rligands)
        self.names = list(map(lambda x : x.split('/')[-1].split('.')[0], self.rligands))

        for i in range(len(self.rligands)):
            self.rligands[i] = self.reset_ligand(LigandPDB.parse(self.rligands[i]))

    def train_ligands(self):
        self.rligands = glob.glob(self.config['random_ligand_folder'] + "/*.pdb") + [self.config['ligand']]
        self.names = list(map(lambda x : x.split('/')[-1].split('.')[0], self.rligands))

        for i in range(len(self.rligands)):
            self.rligands[i] = self.reset_ligand(LigandPDB.parse(self.rligands[i]))
        assert(len(self.rligands) == len(self.names))