from config import config as conf
import glob as glob
from tqdm import tqdm
from rldock.environments.utils import Voxelizer, MultiScorerFromReceptor
from multiprocessing import Pool

def make_receptor( pdb):
    from openeye import oedocking, oechem
    import os.path

    file_name = str(os.path.basename(pdb))
    check_oeb = conf['cache'] + file_name.split(".")[0] + ".oeb"
    if os.path.isfile(check_oeb):

        ifs = oechem.oemolistream(check_oeb)
        ifs.SetFormat(oechem.OEFormat_OEB)
        g = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, g)
        return g
    else:

        proteinStructure = oechem.OEGraphMol()
        ifs = oechem.oemolistream(pdb)
        ofs = oechem.oemolostream(check_oeb)
        ifs.SetFormat(oechem.OEFormat_PDB)
        ofs.SetFormat(oechem.OEFormat_OEB)
        oechem.OEReadMolecule(ifs, proteinStructure)

        box = oedocking.OEBox(*conf['bp_max'], *conf['bp_min'])

        receptor = oechem.OEGraphMol()
        s = oedocking.OEMakeReceptor(receptor, proteinStructure, box)
        assert (s != False)
        oechem.OEWriteMolecule(ofs, receptor)
        ofs.close()
        return receptor

def putincache(pdb_file_name):
    voxelizer = Voxelizer(pdb_file_name, conf, write_cache=True)
    recept = make_receptor(pdb_file_name)
    oe_scorer = MultiScorerFromReceptor(recept)
    return 1

listings = glob.glob(conf['protein_state_folder'] + "*.pdb")

res = []
with Pool(5) as p:
    its = p.imap_unordered(putincache, listings)
    for rest in tqdm(its):
        res.append(rest)