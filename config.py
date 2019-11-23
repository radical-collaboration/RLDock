import os
import math

discrete_trans = 50.0
discrete_theta = 50.0
path = os.path.dirname(os.path.abspath(__file__)) + "/"
config = {
    'action_space_d' : (19.15 / discrete_trans, 19.51 / discrete_trans, 20.8 / discrete_trans),
    'action_space_r' : (2 * math.pi/ discrete_theta, 2 * math.pi / discrete_theta, 2 * math.pi / discrete_theta),
    'protein_wo_ligand' : path + 'resources/lactamase_wo_lignad.pdb',
    'ligand' : path + 'resources/lactamase_ligand.pdb',
    'oe_box' : path +'resources/lacatamase.oeb',
    'bp_dimension': [19.15299892,  19.51399946,  20.80000114],
    'bp_centers' : [ 20.39349937,   7.33399963, -26.39400101],
    'bp_min' : [ 10.81699991,  -2.4230001 , -36.79400158],
    'bp_max' : [ 29.96999884,  17.09099936, -15.99400043],
    'voxelsize' : 0.75,
    'output_size' : (26, 27, 28, 8), # (39,40,42,8),
    'max_steps' : 100,
    'decay' : 0.93, # ^25 = 0.001,
    'random_ligand_folder' :     path + 'resources/rligands',
    'random_ligand_folder_test' : path + 'resources/rligands_eval',
    'voxel_method' : 'C',
    'debug' : False
}
