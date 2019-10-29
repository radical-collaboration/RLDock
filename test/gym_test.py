from pyrosetta import teaching

class RosettaScorer:
    def __init__(self, pdb_file):

        scorefxn = teaching.get_fa_scorefxn()
        print(scorefxn)

rs = RosettaScorer("hi")