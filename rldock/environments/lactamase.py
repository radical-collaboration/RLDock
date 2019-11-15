import copy

import gym
import numpy as np
from gym import spaces
import random
from random import randint
from rldock.environments.LPDB import LigandPDB
from rldock.environments.utils import Scorer, Voxelizer
import glob
import math

# using 6DPT pdb from Lyu et al. (2019, nature)
class LactamaseDocking(gym.Env):
    metadata = {'render.modes': ['human']}

    ## Init the object
    def __init__(self, config ):
        super(LactamaseDocking, self).__init__()
        self.config = config
        # Box space defines the voxel box around the BP. No atom should leave this box. This box is NOT the center.
        self.box_space  = spaces.Box(low=np.array(config['bp_min'], dtype=np.float32) + 5,
                                       high=np.array(config['bp_max'], dtype=np.float32) + 5,
                                       dtype=np.float32)

        self.random_space_init = spaces.Box(low=(-1 * np.max(config['bp_dimension']) / 4.0),
                                            high=(np.max(config['bp_dimension']) / 4.0),
                                            dtype=np.float32,
                                            shape=(3,1))
        self.random_space_rot = spaces.Box(low=0,
                                           high=2 * 3.1415926,
                                           dtype=np.float32,
                                           shape=(3,1))

        lows = -1 * np.array(list(config['action_space_d']) + list(config['action_space_r']), dtype=np.float32)
        highs = np.array(list(config['action_space_d']) + list(config['action_space_r']), dtype=np.float32)
        self.use_random = True
        self.action_space = spaces.Box(low=lows,
                                       high=highs,
                                       dtype=np.float32)
        #tmp file for writing
        self.file = "randoms/" + str(random.randint(0,1000000)) + "_temp.pdb"


        self.reward_range = (-100, np.inf)


        self.observation_space = spaces.Box(low=0, high=1, shape=config['output_size'], #shape=(29, 24, 27, 16),
                                            dtype=np.float32)

        self.voxelizer = Voxelizer(config['protein_wo_ligand'], config)

        self.last_score = 0
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
        self.steps = 0
        self.cur_reward_sum = 0
        self.name = ""
        self.next_exit = False

        self.ro_scorer = None # RosettaScorer(config['protein_wo_ligand'], self.file, self.cur_atom.toPDB()) #takes current action as input, requires reset
        self.oe_scorer = Scorer(config['oe_box']) # takes input as pdb string of just ligand

    def reset_ligand(self, newlig):
        x,y,z  = newlig.get_center()
        return newlig.translate(self.reference_centers[0] - x , self.reference_centers[1] - y, self.reference_centers[2] - z)

    def align_rot(self):
        for i in range(3):
            self.rot[i] = self.rot[i] % (2 * 3.14159265)
            if self.rot[i] < 0:
                self.rot[i] = 2*3.14159265 + self.rot[i]

    def decay_action(self, action, just_trans=False):
        for i in range(3 if just_trans else len(action)):
            action[i] *= math.pow(self.config['decay'], self.steps)
        return action

    def get_action(self, action):
        # for i in range(3):
        #     action[i] *= 1
        # for i in [3,4,5]:
        #     action[i] /= 1.59154
        return action

    def step(self, action):
        if random.random() < 0.01:
            print(action)
        if np.any(np.isnan(action)):
            print(action)
            print("ERROR, nan action from get action")
            exit()

        action = self.get_action(action)
        action = self.decay_action(action)
        self.trans[0] += action[0]
        self.trans[1] += action[1]
        self.trans[2] += action[2]
        self.rot[0] += action[3]
        self.rot[1] += action[4]
        self.rot[2] += action[5]

        self.cur_atom = self.cur_atom.translate(action[0], action[1], action[2])
        self.cur_atom = self.cur_atom.rotate(action[3], action[4], action[5])
        self.align_rot()

        oe_score = self.oe_scorer(self.cur_atom.toPDB())
        # ro_score = self.ro_scorer(*action) #rosetta pass trans
        reset = self.decide_reset(oe_score)

        self.last_score = oe_score
        self.steps += 1

        reward = self.get_reward_from_ChemGauss4(oe_score, reset)

        if reset and self.ro_scorer is not None:
            reward += self.ro_scorer(*self.trans)

        self.cur_reward_sum += reward

        if self.next_exit:
            self.next_exit = False
            return self.get_obs(), reward, True, {}

        if reset:
            self.next_exit = True

        obs = self.get_obs()
        if np.any(np.isnan(obs)):
            print(obs)
            print("ERROR, nan action from get obs")

        return obs,\
               reward,\
               reset, \
               {}

    def decide_reset(self, score):
         return self.steps > self.config['max_steps'] or (self.steps > 10 and not self.check_atom_in_box())

    def get_reward_from_ChemGauss4(self, score, reset=False):
        if reset:
            return np.clip(np.array(score * -1), 0, 10)  * 1
        else:
            return np.clip(np.array(score * -1), 0, 10)  * 0.01

    def reset(self, random=False, many_ligands =True):
        if many_ligands and self.rligands != None and self.use_random:
            idz = randint(0, len(self.rligands) - 1)
            start_atom = copy.deepcopy(self.rligands[idz])
            self.name = self.names[idz]

        elif many_ligands and self.rligands != None :
            start_atom = copy.deepcopy(self.rligands.pop(0))
            self.name = self.names.pop(0)
        else:
            start_atom = copy.deepcopy(self.atom_center)

        if random:
            x,y,z, = self.random_space_init.sample().flatten().ravel()
            x_theta, y_theta, z_theta = self.random_space_rot.sample().flatten().ravel()
            self.trans = [x,y,z]
            self.rot = [x_theta, y_theta, z_theta]
            random_pos = start_atom.translate(x,y,z)
            random_pos = random_pos.rotate(theta_x=x_theta, theta_y=y_theta, theta_z=z_theta)
        else:
            self.trans = [0,0,0]
            self.rot   = [0,0,0]
            random_pos = start_atom

        if self.ro_scorer is not None:
            self.ro_scorer.reset(random_pos.toPDB())
        self.cur_atom = random_pos
        self.last_score = self.oe_scorer(self.cur_atom.toPDB())
        self.steps = 0
        self.cur_reward_sum=0
        return self.get_obs()

    def get_obs(self):
        return self.voxelizer(self.cur_atom.toPDB()).squeeze(0)

    def render(self, mode='human'):
        print("Score", self.last_score, self.cur_reward_sum)
        return self.cur_atom, self.name

    def close(self):
        pass

    def check_atom_in_box(self):
        ans = True
        for atom in self.cur_atom.hetatoms:
            ans &= self.box_space.contains([atom.x_ortho_a, atom.y_ortho_a, atom.z_ortho_a])
        return ans

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