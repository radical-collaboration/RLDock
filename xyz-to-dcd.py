import MDAnalysis


u = MDAnalysis.coordinates.XYZ.XYZReader("test.xyz")
print(u.n_atoms, u.totaltime, u.n_frames)
with MDAnalysis.coordinates.TRJ.TRJ("protein.dcd", u.n_atoms) as W:
    for ts in u.trajectory:
        print(ts)