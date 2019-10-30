import copy

import gym
import numpy as np
from gym import spaces
import random
from rldock.environments.LPDB import LigandPDB
from rldock.environments.pdb_utiils import CenterPDB
from rldock.environments.utils import Scorer, Voxelizer, RosettaScorer


# using 6DPT pdb from Lyu et al. (2019, nature)
class LactamaseDocking(gym.Env):
    metadata = {'render.modes': ['human']}

    ## Init the object
    def __init__(self, config ):
        super(LactamaseDocking, self).__init__()

        # Box space defines the voxel box around the BP. No atom should leave this box. This box is NOT the center.
        self.box_space  = spaces.Box(low=np.array(config['bp_min'], dtype=np.float32),
                                       high=np.array(config['bp_max'], dtype=np.float32),
                                       dtype=np.float32)

        self.action_space = spaces.Box(low=-1 * np.array(3 * [config['action_space_d']], dtype=np.float32),
                                       high=np.array(3 * [config['action_space_d']], dtype=np.float32),
                                       dtype=np.float32)
        #tmp file for writing
        self.file = "randoms/" + str(random.randint(0,1000000)) + "_temp.pdb"


        self.reward_range = (-100, np.inf)


        self.observation_space = spaces.Box(low=0, high=1, shape=config['output_size'], #shape=(29, 24, 27, 16),
                                            dtype=np.float32)

        self.voxelizer = Voxelizer(config['protein_wo_ligand'])

        self.last_score = 0
        self.atom_center =  LigandPDB.parse(config['ligand'])
        self.cur_atom = copy.deepcopy(self.atom_center)
        self.trans = [0,0,0]
        self.steps = 0
        self.cur_reward_sum = 0

        self.ro_scorer = RosettaScorer(config['protein-wo_ligand'], self.file, self.cur_atom.toPDB()) #takes current action as input, requires reset
        self.oe_scorer = Scorer(config['protein-wo_ligand']) # takes input as pdb string of just ligand

    def align_rot(self):
        for i in range(3):
            self.rot[i] = self.rot[i] % 360
            if self.rot[i] < 0:
                self.rot[i] = 360 + self.rot[i]


    def step(self, action):
        self.trans[0] += action[0]
        self.trans[1] += action[1]
        self.trans[2] += action[2]

        # self.rot[0] += action[3]
        # self.rot[1] += action[4]
        # self.rot[2] += action[5]
        # self.align_rot()

        self.cur_atom = self.cur_atom.translate(action[0], action[1], action[2])
        # self.cur_atom = self.cur_atom.rotate(action[3], action[4], action[5])
        oe_score = self.oe_scorer(self.cur_atom.toPDB())
        # ro_score = self.ro_scorer(*action) #rosetta pass trans
        reset = self.decide_reset(oe_score)

        self.last_score = oe_score
        self.steps += 1

        reward = self.get_reward_from_ChemGauss4(oe_score)
        if reset:
            reward += self.ro_scorer(*self.trans)

        self.cur_reward_sum += reward
        return self.get_obs(),\
               reward,\
               reset, \
               {}

    def decide_reset(self, score):
         return self.steps > 100 or (not self.check_atom_in_box())


    def get_reward_from_ChemGauss4(self, score):
        return np.clip(np.array(score * -1), -100, 10000)

    def reset(self, random=False):
        if random:
            x,y,z = self.action_space.sample().flatten().ravel()
            #
            self.trans = [x,y,z]
            #
            #
            random_pos = self.atom_center.translate(x,y,z)
        else:
            self.trans = [0,0,0]
            self.rot = [0,0,0]
            random_pos = copy.deepcopy(self.atom_center)

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