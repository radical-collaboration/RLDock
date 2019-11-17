import math
config = {
    'action_space_d' : (5, 5, 5),
    'action_space_r' : (0.96, 0.96, 0.96),
    'protein_wo_ligand' : 'resources/lactamase_wo_lignad.pdb',
    'ligand' : 'resources/lactamase_ligand.pdb',
    'oe_box' : 'resources/lacatamase.oeb',
    'bp_dimension': [19.15299892,  19.51399946,  20.80000114],
    'bp_centers' : [ 20.39349937,   7.33399963, -26.39400101],
    'bp_min' : [ 10.81699991,  -2.4230001 , -36.79400158],
    'bp_max' : [ 29.96999884,  17.09099936, -15.99400043],
    'voxelsize' : 0.5,
    'output_size' : (40,40,42,8),
    'max_steps' : 100,
    'decay' : 0.01, # ^25 = 0.001,
    'random_ligand_folder' :      'resources/rligands',
    'random_ligand_folder_test' : 'resources/rligands_eval',
    'voxel_method' : 'numba'
}
