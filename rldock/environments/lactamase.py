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
    def __init__(self):
        super(LactamaseDocking, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects

        self.box_space  = spaces.Box(low=np.array( [-5, -5, -5, -0, -0, -0], dtype=np.float32),
                                       high=np.array([5, 5, 5,  0,  0,  0], dtype=np.float32),
                                       dtype=np.float32)

        # self.start_space  = spaces.Box(low=np.array( [-28.9, -24, -26, -180, -180, -180], dtype=np.float32),
        #                                high=np.array([28.9,   24,  26,  180,  180,  180], dtype=np.float32),
        #                                dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-0.5, -0.5,  -0.5, -0, -0, -0], dtype=np.float32),
                                       high=np.array([0.5,  0.5,   0.5,  0,  0,  0], dtype=np.float32),
                                       dtype=np.float32)
        self.file = "randoms/" + str(random.randint(0,1000000)) + "_temp.pdb"
        self.reward_range = (-100, np.inf)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(20, 16, 18, 16), #shape=(29, 24, 27, 16),
                                            dtype=np.float32)

        atom = LigandPDB.parse("resources/ligand.pdb")
        self.voxelizer = Voxelizer('resources/center_protein_wo_ligand.pdb')

        cb = CenterPDB(to_x=-18.942, to_y=-2.829, to_z=19.666)
        #range -10.045, 47.93
        #range -20.959, 26.6
        #range -46.273, 6.9

        cb.fit(atom)
        self.last_score = 0
        self.atom_center = cb.transform(atom)
        self.cur_atom = copy.deepcopy(self.atom_center)
        self.trans = [0,0,0]
        self.rot   = [0,0,0]
        self.steps = 0
        self.scorer = RosettaScorer("resources/center_protein.pdb", self.file, self.cur_atom.toPDB())


    def align_rot(self):
        for i in range(3):
            self.rot[i] = self.rot[i] % 360
            if self.rot[i] < 0:
                self.rot[i] = 360 + self.rot[i]


    def step(self, action):
        self.trans[0] += action[0]
        self.trans[1] += action[1]
        self.trans[2] += action[2]

        self.rot[0] += action[3]
        self.rot[1] += action[4]
        self.rot[2] += action[5]
        self.align_rot()

        self.cur_atom = self.cur_atom.translate(action[0], action[1], action[2])
        self.cur_atom = self.cur_atom.rotate(action[3], action[4], action[5])
        score = self.scorer(*action)
        reset = self.decide_reset(score)

        self.last_score = score
        self.steps += 1

        return self.get_obs(),\
               self.get_reward_from_ChemGauss4(score),\
               reset, \
               {}

    def decide_reset(self, score):
         return self.steps > 100 or (not self.check_atom_in_box())


    def get_reward_from_ChemGauss4(self, score):
        return np.clip(np.array(score * -1), -100, 10000)

    def reset(self, random=False):
        if random:
            x,y,z,theta_x, theta_y, theta_z = self.action_space.sample().flatten().ravel()
            #
            self.trans = [x,y,z]
            self.rot   = [theta_x, theta_y, theta_z]
            self.align_rot()
            #
            #
            random_pos = self.atom_center.translate(x,y,z)
            random_pos = random_pos.rotate(theta_x, theta_y, theta_z)
        else:
            self.trans = [0,0,0]
            self.rot = [0,0,0]
            random_pos = copy.deepcopy(self.atom_center)

        # random_pos = copy.deepcopy(self.atom_center)
        self.scorer.reset(random_pos.toPDB())
        score = self.scorer(self.trans[0], self.trans[1], self.trans[2], self.rot[0], self.rot[1], self.rot[2])
        self.cur_atom = random_pos
        self.last_score = score
        self.steps = 0
        return self.get_obs()

    def get_obs(self):
        return self.voxelizer(self.cur_atom.toPDB()).squeeze(0)

    def render(self, mode='human'):
        print("Score", self.last_score)
        return self.cur_atom

    def close(self):
        pass

    def check_atom_in_box(self):
        ans = self.box_space.contains([self.trans[0], self.trans[1], self.trans[2], 0,0 ,0])
        # for atom in self.cur_atom.hetatoms:
        #     ans &= self.box_space.contains([atom.x_ortho_a, atom.y_ortho_a, atom.z_ortho_a, 0 ,0 ,0])
        return ans