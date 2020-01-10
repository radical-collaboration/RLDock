from rldock.environments import LPDB
from config import config

lig = LPDB.LigandPDB.parse("/PycharmProjects/RLDock/drugscreen/newpdbs/0.pdb")

centers = lig.get_center()
lig = lig.translate(*centers)

