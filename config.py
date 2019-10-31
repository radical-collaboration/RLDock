config = {
    'action_space_d' : 2.5,
    'protein_wo_ligand' : 'resources/lactamase_wo_lignad.pdb',
    'ligand' : 'resources/lactamase_ligand.pdb',
    'oe_box' : 'resources/lacatamase.oeb',
    'bp_dimension': [19.15299892,  19.51399946,  20.80000114],
    'bp_centers' : [ 20.39349937,   7.33399963, -26.39400101],
    'bp_min' : [ 10.81699991,  -2.4230001 , -36.79400158],
    'bp_max' : [ 29.96999884,  17.09099936, -15.99400043],
    'voxelsize' : 1,
    'output_size' : (20,20,21,16),
    'max_steps' : 25,
    'decay' : 0.78 # ^25 = 0.001
}
