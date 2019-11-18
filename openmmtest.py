from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

pdb = app.PDBFile('resources/lactamase_with_ligand.pdb')

forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
unmatched_residues = forcefield.getUnmatchedResidues(pdb.topology)
print(unmatched_residues)

system = forcefield.createSystem(topology=pdb.topology, implicitSolvent=app.GBn2,
                                 nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.0 * unit.nanometers,
                                 constraints=app.HBonds,
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)

integrator = mm.LangevinIntegrator(300 * unit.kelvin,
                                   1.0 / unit.picoseconds,
                                   2.0 * unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)
# TODO: This should just recognize whatever the computer is capable of, not force CUDA.
platform = mm.Platform.getPlatformByName('OpenCL')
# TODO: I am not sure if mixed precision is necessary. It dramatically changes the results.
# properties = {'CudaPrecision': 'mixed'}

simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

simulation.minimizeEnergy()
energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
    unit.kilojoule / unit.mole)
print(energy)