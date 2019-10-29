import copy
from rldock.environments import LPDB

class CenterPDB:
    def __init__(self, translate=True, to_x=0, to_y=0, to_z=0):
        self.translate = translate
        self.translate_x = to_x
        self.translate_y = to_y
        self.translate_z = to_z

        self.to_x = to_x
        self.to_y = to_y
        self.to_z = to_z

    def fit(self, pdb : LPDB.LigandPDB):
        # c = 0
        # for atom in pdb.hetatoms:
        #     c += 1
        #     self.translate_x += atom.x_ortho_a
        #     self.translate_y += atom.y_ortho_a
        #     self.translate_z += atom.z_ortho_a
        #
        # # computed mean direction
        # self.translate_x /= c
        # self.translate_y /= c
        # self.translate_z /= c
        #
        # #move center
        #
        # #turn to 0,0,0
        # self.translate_x = self.to_x - self.translate_x
        # self.translate_y = self.to_y - self.translate_y
        # self.translate_z = self.to_z - self.translate_z
        pass

    # returns copy
    def transform(self, pdb : LPDB.LigandPDB):
        # for each atom, translate by shape
        pdb_c = copy.deepcopy(pdb)

        for atom in pdb_c.hetatoms:
            atom.x_ortho_a += self.translate_x
            atom.y_ortho_a += self.translate_y
            atom.z_ortho_a += self.translate_z

        return pdb_c