from openeye import oechem, oedocking
import numpy as np
from rdkit.Chem import AllChem
from rldock.environments import LPDB
import scipy

class RosettaScorer:
    def __init__(self, pdb_file, rf, tw):
        import pyrosetta
        from pyrosetta import teaching
        import pyrosetta.rosetta.numeric
        self.rose = pyrosetta.rosetta.numeric

        self.rf = rf
        pyrosetta.init()
        with open(pdb_file, 'r') as f:
            self.prior_detail = "".join(f.readlines()[:-1]) # strip off end.
        self.ligand_maker = pyrosetta.pose_from_pdb
        self.score  = teaching.get_fa_scorefxn()
        self.reset(tw)

    def reset(self, pdb):
        with open(self.rf, 'w') as f:
            f.write(self.prior_detail)
            f.write(pdb)
        self.pose = self.ligand_maker(self.rf)

    def __call__(self, x_pos, y_pos, z_pos):
        x = self.rose.xyzMatrix_double_t()
        # #row1
        x.xx, x.xy, x.xz, x.yx, x.yy, x.yz, x.zx, x.zy, x.zz = scipy.identity(3).flatten().ravel()

        v = self.rose.xyzVector_double_t()
        v.x = x_pos
        v.y = y_pos
        v.z = z_pos

        self.pose.residue(len(self.pose.sequence())).apply_transform_Rx_plus_v(x,v)
        return self.score(self.pose)

## Basic scorer, loads pdb from file

class MultiScorer:
    def __init__(self, pdb_file):
        self.receptor = oechem.OEGraphMol()
        self.scorers = [oedocking.OEScore(oedocking.OEScoreType_Shapegauss),
                        oedocking.OEScore(oedocking.OEScoreType_Chemscore),
                        oedocking.OEScore(oedocking.OEScoreType_Chemgauss3),
                        oedocking.OEScore(oedocking.OEScoreType_Chemgauss4),
                        ]
        self.score = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        oedocking.OEReadReceptorFile(self.receptor, pdb_file)
        for score in self.scorers:
            score.Initialize(self.receptor)

    def __call__(self, item : str):
        ligand = oechem.OEGraphMol()
        ligand_name = oechem.oemolistream()
        ligand_name.openstring(item)
        oechem.OEReadPDBFile(ligand_name, ligand)

        return [scorer.ScoreLigand(ligand) for scorer in self.scorers]

class Scorer:

    def __init__(self, pdb_file):
        self.receptor = oechem.OEGraphMol()
        self.score = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        oedocking.OEReadReceptorFile(self.receptor, pdb_file)
        self.score.Initialize(self.receptor)

    def __call__(self, item : str):
        ligand = oechem.OEGraphMol()
        ligand_name = oechem.oemolistream()
        ligand_name.openstring(item)
        oechem.OEReadPDBFile(ligand_name, ligand)
        return self.score.ScoreLigand(ligand)


'''
This class will consider a 3D ligand, and only consider translation and rotation
'''
class RigidLigand:
    def __init__(self, pdb_file):
        self.ligand = LPDB.LigandPDB.parse(pdb_file)

class MinMax:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max
        self.eps = 1e10

    def __call__(self):
        return self.min, self.max

    def update(self, x):
        if self.min is None:
            self.min = x
        else:
            self.min = min(self.min, x)

        if self.max is None and (x <= self.eps):
            self.max = x
        elif (x <= self.eps):
            self.max = max(self.max, x)

def l2_action(action):
    l2 = np.sum(np.power(np.array(action),2))
    return float(l2)

from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from moleculekit.smallmol.smallmol import SmallMol

class Voxelizer:

    def __init__(self, pdb_structure, config):
        from moleculekit.molecule import Molecule
        from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
        self.config = config
        prot = Molecule(pdb_structure)

        prot.write("/Users/austin/gpcr/test_out.pdb")

        prot = prepareProteinForAtomtyping(prot, verbose=False)
        prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, buffer=0, voxelsize=config['voxelsize'], boxsize=config['bp_dimension'],
                                                     center=config['bp_centers'], validitychecks=False)
        nchannels = prot_vox.shape[1]

        self.prot_vox_t = prot_vox.transpose().reshape([1, nchannels, prot_N[0], prot_N[1], prot_N[2]])


    def __call__(self, lig_pdb, quantity='all'):

        slig = SmallMol(AllChem.MolFromPDBBlock(lig_pdb, sanitize=True, removeHs=False))
        lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, buffer=0, voxelsize=self.config['voxelsize'], boxsize=self.config['bp_dimension'],
                                                     center=self.config['bp_centers'], validitychecks=False, method=self.config['voxel_method'])
        nchannels = lig_vox.shape[1]
        if quantity == 'all':
            x = lig_vox.transpose().reshape([1, nchannels, lig_N[0], lig_N[1], lig_N[2]]) + self.prot_vox_t
        elif quantity == 'ligand':
            x = lig_vox.transpose().reshape([1, nchannels, lig_N[0], lig_N[1], lig_N[2]])
        else:
            x = self.prot_vox_t


        return np.transpose(np.concatenate([ x], axis=1), (0,2,3,4,1))
