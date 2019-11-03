import copy

import gym
import numpy as np
from gym import spaces
import random
from rldock.environments.LPDB import LigandPDB
from rldock.environments.pdb_utiils import CenterPDB
from rldock.environments.utils import Scorer, Voxelizer, RosettaScorer

import math
# using 6DPT pdb from Lyu et al. (2019, nature)
class LactamaseDocking(gym.Env):
    metadata = {'render.modes': ['human']}

    ## Init the object
    def __init__(self, config ):
        super(LactamaseDocking, self).__init__()
        self.config = config
        # Box space defines the voxel box around the BP. No atom should leave this box. This box is NOT the center.
        self.box_space  = spaces.Box(low=np.array(config['bp_min'], dtype=np.float32),
                                       high=np.array(config['bp_max'], dtype=np.float32),
                                       dtype=np.float32)

        self.random_space_init = spaces.Box(low=(-1 * np.max(config['bp_dimension']) / 2.0), high=(np.max(config['bp_dimension']) / 2.0), dtype=np.float32, shape=(3,1))

        self.action_space = spaces.Box(low=-1 * np.array(3 * [config['action_space_d']], dtype=np.float32),
                                       high=np.array(3 * [config['action_space_d']], dtype=np.float32),
                                       dtype=np.float32)
        #tmp file for writing
        self.file = "randoms/" + str(random.randint(0,1000000)) + "_temp.pdb"


        self.reward_range = (-100, np.inf)


        self.observation_space = spaces.Box(low=0, high=1, shape=config['output_size'], #shape=(29, 24, 27, 16),
                                            dtype=np.float32)

        self.voxelizer = Voxelizer(config['protein_wo_ligand'], config)

        self.last_score = 0
        self.atom_center =  LigandPDB.parse(config['ligand'])
        self.cur_atom = copy.deepcopy(self.atom_center)
        self.trans = [0,0,0]
        self.steps = 0
        self.cur_reward_sum = 0

        self.ro_scorer = None # RosettaScorer(config['protein_wo_ligand'], self.file, self.cur_atom.toPDB()) #takes current action as input, requires reset
        self.oe_scorer = Scorer(config['oe_box']) # takes input as pdb string of just ligand

    def align_rot(self):
        for i in range(3):
            self.rot[i] = self.rot[i] % 360
            if self.rot[i] < 0:
                self.rot[i] = 360 + self.rot[i]

    def decay_action(self, action):
        for i in range(len(action)):
            action[i] *= math.pow(self.config['decay'], self.steps)
        return action

    def step(self, action):
        action = self.decay_action(action)
        self.trans[0] += action[0]
        self.trans[1] += action[1]
        self.trans[2] += action[2]


        self.cur_atom = self.cur_atom.translate(action[0], action[1], action[2])
        # self.cur_atom = self.cur_atom.rotate(action[3], action[4], action[5])
        oe_score = self.oe_scorer(self.cur_atom.toPDB())
        # ro_score = self.ro_scorer(*action) #rosetta pass trans
        reset = self.decide_reset(oe_score)

        self.last_score = oe_score
        self.steps += 1

        reward = self.get_reward_from_ChemGauss4(oe_score, reset)

        if reset and self.ro_scorer is not None:
            reward += self.ro_scorer(*self.trans)

        self.cur_reward_sum += reward
        return self.get_obs(),\
               reward,\
               reset, \
               {}

    def decide_reset(self, score):
         return self.steps > self.config['max_steps'] or (not self.check_atom_in_box())

    def get_reward_from_ChemGauss4(self, score, reset=False):
        if reset:
            return np.clip(np.array(score * -1), -1, 10000)  * 5
        else:
            return np.clip(np.array(score * -1), -1, 1)  * 0.01


    def reset(self, random=True):
        if random:
            x,y,z = self.random_space_init.sample().flatten().ravel() * 0.5
            self.trans = [x,y,z]
            random_pos = self.atom_center.translate(x,y,z)
        else:
            self.trans = [0,0,0]
            self.rot   = [0,0,0]
            random_pos = copy.deepcopy(self.atom_center)

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
        return self.cur_atom

    def close(self):
        pass

    def check_atom_in_box(self):
        ans = True
        for atom in self.cur_atom.hetatoms:
            ans &= self.box_space.contains([atom.x_ortho_a, atom.y_ortho_a, atom.z_ortho_a])
        return ans