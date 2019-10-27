from openeye import oechem, oedocking
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rldock.environments import LPDB, pdb_utiils

# self.ligand = oechem.OEGraphMol()
# ligand_name = oechem.oemolistream("ligand.pdb")
# oechem.OEReadPDBFile(ligand_name, ligand)
#     print(score.ScoreLigand(ligand))



## Basic scorer, loads pdb from file
class Scorer:

    def __init__(self, pdb_file):
        self.receptor = oechem.OEGraphMol()
        self.score = oedocking.OEScore(oedocking.OEScoreType_Chemscore)
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


from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from moleculekit.smallmol.smallmol import SmallMol

class Voxelizer:
    x_center = 18.945
    y_center = 2.820
    z_center = -19.666

    x_size = 57.975
    y_size = 47.559
    z_size = 53.214

    def __init__(self, pdb_structure):
        from moleculekit.molecule import Molecule
        from moleculekit.tools.atomtyper import prepareProteinForAtomtyping

        prot = Molecule(pdb_structure)
        prot = prepareProteinForAtomtyping(prot)
        prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, buffer=0.5, voxelsize=3, boxsize=[self.x_size,self.y_size,self.z_size],
                                                     center=[self.x_center, self.y_center, self.z_center], validitychecks=False)
        nchannels = prot_vox.shape[1]

        self.prot_vox_t = prot_vox.transpose().reshape([1, nchannels, prot_N[0], prot_N[1], prot_N[2]])


    def __call__(self, lig_pdb):
        slig = SmallMol(AllChem.MolFromPDBBlock(lig_pdb))
        lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, buffer=0.5, voxelsize=3,  boxsize=[self.x_size,self.y_size,self.z_size],
                                                     center=[self.x_center, self.y_center, self.z_center], validitychecks=False)
        nchannels = lig_vox.shape[1]
        lig_vox_t = lig_vox.transpose().reshape([1, nchannels, lig_N[0], lig_N[1], lig_N[2]])
        return np.transpose(np.concatenate([self.prot_vox_t, lig_vox_t], axis=1), (0,2,3,4,1))
