from unittest import TestCase


class TestLigandAtom(TestCase):
    def test_fromPDB(self):
        from rldock.environments.LPDB import LigandAtomParser, LigandAtom, LigandPDB
        from rldock.environments.pdb_utiils import CenterPDB
        from rldock.environments.utils import Scorer

        # with open("/PycharmProjects/RLDOCK/testing/ligand.pdb") as f:
        #     row = f.readlines()[

        atom = LigandPDB.parse("/PycharmProjects/RLDOCK/testing/ligand.pdb")
        cb = CenterPDB(to_x=18.9425, to_y = 2.82, to_z=-19.66)
        cb.fit(atom)
        atom_c = cb.transform(atom)
        print(cb.translate_x, cb.translate_y, cb.translate_z)
        sc = Scorer("/PycharmProjects/RLDOCK/testing/6DPT_receptor.oeb.gz")
        print(sc(atom.toPDB()))
        print(atom_c.toPDB())

        with open("/PycharmProjects/RLDock/test_center.pdb", 'w') as f:
            f.write(atom_c.toPDB())

