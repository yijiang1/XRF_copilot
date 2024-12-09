#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yi Jiang
Simulate 2D XRF projection from 3D elemental objects
Based on Panpan's FL_signal_reprojection_fn.py script
"""

import os
import sys
import datetime
import numpy as np
import h5py
#from mpi4py import MPI
import xraylib as xlib
import xraylib_np as xlib_np
import torch as tc
tc.set_default_dtype(tc.float32)  # Set the default tensor dtype
#tc.set_default_device('cuda:0')  # Set the default device to CPU (or 'cuda:0' for GPU)
import time
from util import rotate, prepare_fl_lines, intersecting_length_fl_detectorlet
from misc import print_flush_root, create_summary
from forward_model import PPM
import warnings
from mendeleev import element

warnings.filterwarnings("ignore")

def simulate_XRF_maps(base_path, ground_truth_file, output_file_base,
                      incident_probe_intensity, probe_energy, 
                      model_probe_attenuation, model_self_absorption,
                      element_lines_roi, 
                      sample_size_cm, det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                      batch_size,
                      P_file='',
                      overwrite_P=False,
                      gpu_id=0):

    params = locals()
    if tc.cuda.is_available() and gpu_id >= 0:  
        dev = tc.device('cuda:{}'.format(gpu_id))
    else:  
        dev = "cpu"
    print(f'Device: {dev}')
    ####----------------------------------------------------------------------------------####
    #### load true 3D objects ####
    X = np.load(f'{base_path}/{ground_truth_file}')
    print(f'Test objects size: {X.shape}')
    X = tc.from_numpy(X).float().to(dev) #cpu

    sample_size_n = X.shape[1]
    sample_height_n = X.shape[3]

    dia_len_n = int(1.2 * (sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5) #number of voxels along the diagonal direction (0,0,0) -> (nx, ny, nz)
    n_voxel_batch = batch_size * sample_size_n #number of voxels in each batch
    n_voxel = sample_height_n * sample_size_n**2     # total number of voxels. 
    # sample_size_n seems to be the size along x and y axis in Figure 5.1.
    # sample_height_n seems to be the size along z axis in Figure 5.1.

    ####----------------------------------------------------------------------------------####
    #### parallelization #### 
    n_ranks = 1
    rank = 0

    minibatch_ls_0 = tc.arange(n_ranks).to(dev) #dev
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * batch_size)
    print(f'Number of batches: {n_batch}')

    ####----------------------------------------------------------------------------------####
    #### get physical constants for the input elements and energy #### 

    # Figure out the number of elements in from element_lines_roi
    element_names = np.unique(element_lines_roi[:, 0])
    n_element = len(element_names)
    # Get atomic number for each element
    atomic_numbers = {name: element(name).atomic_number for name in element_names}
    # Sort the list based on atomic numbers
    atomic_numbers = dict(sorted(atomic_numbers.items(), key=lambda item: item[1]))
    element_names = sorted(element_names, key=lambda name: atomic_numbers[name])
    print(f'Atomic_numbers: {atomic_numbers}')   
    sorted_indices = np.argsort([atomic_numbers[element] for element in element_lines_roi[:, 0]])
    element_lines_roi = element_lines_roi[sorted_indices]
    # Figure out the number of FL lines for each element
    n_line_group_each_element = np.array([np.sum(element_lines_roi[:, 0] == name) for name in element_names])
    print(n_line_group_each_element)
    # Create a lookup table of the fluorescence lines of interests
    fl_all_lines_dic = prepare_fl_lines(element_lines_roi,                           
                                        n_line_group_each_element, probe_energy, 
                                        sample_size_n, sample_size_cm) #cpu
    
    detected_fl_unit_concentration = tc.as_tensor(fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(dev)

    # Get the mass attenuation cross section for each XRF line (3rd row in Table 5.3.1) as a list ####
    mass_attenuation_cross_section_FL = tc.as_tensor(xlib_np.CS_Total(np.array(list(atomic_numbers.values())),
                                                    fl_all_lines_dic["fl_energy"])).float().to(dev) #dev
    
    n_line_group_each_element = tc.IntTensor(fl_all_lines_dic["n_line_group_each_element"]).to(dev)
    print(n_line_group_each_element)

    n_lines = fl_all_lines_dic["n_lines"] #scalar
    print(f'Total number of energy lines (n_lines)={n_lines}')

    # Create the elements list using element_lines_roi
    channel_name_roi_ls = np.array([
        element_line_roi[0] if element_line_roi[1] == "K" 
        else f"{element_line_roi[0]}_{element_line_roi[1]}"
        for element_line_roi in element_lines_roi
    ]).astype("S5")  
    #print(f'channel_name_roi_ls:')
    #print(channel_name_roi_ls) # this format is for xrf maps??
    scaler_names = np.array(["place_holder", "us_dc", "ds_ic", "abs_ic"]).astype("S12")

    # Calculate the mass attenuation cross section of probe (2nd row in Table 5.3.1) as a list ####
    probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(np.array(list(atomic_numbers.values())), probe_energy).flatten()).to(dev)

    #### det_solid_angle_ratio is used only for simulated dataset (use_std_calibation: False, manual_det_area: False, manual_det_coord: False)
    #### in which the incident probe intensity is not calibrated with the axo_std file.
    #### The simulated collected XRF photon number is estimated by multiplying the generated
    #### fluorescence photon number by "det_solid_angle_ratio" to account for the limited solid angle and the detecting efficiency of the detector
    
#         #### Calculate the detecting solid angle covered by the area of the spherical cap covered by the detector #### 
#         #### OPTION A: estimate the solid angle by the curved surface
#         # The distance from the sample to the boundary of the detector
#         r = (det_from_sample_cm**2 + (det_dia_cm/2)**2)**0.5   
#         # The height of the cap
#         h =  r - det_from_sample_cm
#         # The area of the cap area
#         fl_sig_collecting_cap_area = np.pi*((det_dia_cm/2)**2 + h**2)
#         # The ratio of the detecting solid angle / full soilid angle
#         det_solid_angle_ratio = fl_sig_collecting_cap_area / (4*np.pi*r**2)

    #### OPTION B: estimate the solid angle by the flat surface
    det_solid_angle_ratio = (np.pi * (det_size_cm/2)**2) / (4*np.pi * det_from_sample_cm**2)
    #print(f'det_solid_angle_ratio={det_solid_angle_ratio}')

    #### signal_attenuation_factor is used to account for other factors that cause the attenuation of the XRF
    #### except for the limited solid angle and self-absorption
    signal_attenuation_factor = 1.0
    #print(f'signal_attenuation_factor={signal_attenuation_factor}')

    ####----------------------------------------------------------------------------------####
    #### get P array #### 
    if P_file is None or P_file.strip() == "":  #use default file name
        P_file = f'P_det_size{det_size_cm}_spacing_{det_size_cm}_dist{det_from_sample_cm}_sample_size{sample_size_cm}_nxy{sample_size_n}_nz{sample_height_n}'
    P_save_path = os.path.join(base_path, P_file)

    #Check if the P array exists, if it doesn't exist, call the function to calculate the P array and store it as a .h5 file.
    if not os.path.isfile(P_save_path + ".h5") or overwrite_P:
        print(f'Calculating the intersecting length array P. This will take quite some time...')
        intersecting_length_fl_detectorlet(n_ranks, rank,
                                           det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                           sample_size_n, sample_size_cm, sample_height_n, 
                                           base_path, P_file) #has to use CPU for this step
        print(f'Completed. P is saved at {P_save_path}.h5')
    else:
        print(f'Loading an existing intersecting length array P from {P_save_path}.h5')

    P_handle = h5py.File(P_save_path + ".h5", 'r')

    ####----------------------------------------------------------------------------------####
    #### I/O #### 
    #stdout_options = {'root':0, 'output_folder': base_path, 'save_stdout': True, 'print_terminal': False}
    stdout_options = {'root':0, 'output_folder': base_path, 'save_stdout': False, 'print_terminal': True}
    timestr = str(datetime.datetime.today())     
    print_flush_root(0, val=f"time: {timestr}", output_file='', **stdout_options)
    
    sim_XRF_file = f'{base_path}/{output_file_base}_xrf'
    sim_XRT_file = f'{base_path}/{output_file_base}_xrt'
    params_file_name = f'{output_file_base}_params'

    suffixes = []
    if model_probe_attenuation:
        suffixes.append('pa')
    if model_self_absorption:
        suffixes.append('sa')
    sim_XRF_file += '_' + '_'.join(suffixes) + '.h5' if suffixes else '.h5'
    sim_XRT_file += '_' + '_'.join(suffixes) + '.h5' if suffixes else '.h5'
    params_file_name += '_' + '_'.join(suffixes) + '.txt' if suffixes else '.txt'

    # initialize h5 files for saving simulated signals
    with h5py.File(sim_XRF_file, 'w') as d:
        grp = d.create_group("exchange")
        data = grp.create_dataset("data", shape=(n_lines, sample_height_n, sample_size_n), dtype="f4")
        elements = grp.create_dataset("elements", data = channel_name_roi_ls)

    with h5py.File(sim_XRT_file, 'w') as d:
        grp = d.create_group("exchange")
        data = grp.create_dataset("data", shape=(4, sample_height_n, sample_size_n), dtype="f4")

    params['P_file'] = P_file
    params['element_lines_roi'] = element_lines_roi
    create_summary(base_path, params, fname=params_file_name)

    ####----------------------------------------------------------------------------------####
    #### simulation ####
    start_time_total = datetime.datetime.now()  # Start time for the simulation
    theta = tc.tensor(0.0) #rotation angle in tomography
    ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
    if model_self_absorption == True:
        X_ap_rot = rotate(X, theta, dev) #dev
        lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * mass_attenuation_cross_section_FL.view(n_element, n_lines, 1, 1) #dev #Eq. 5.9
        lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev
    else:
        lac = 0.

    for m in range(n_batch):
        start_time = datetime.datetime.now()  # Start time for the iteration
        minibatch_ls = n_ranks * m + minibatch_ls_0  #dev, e.g. [5,6,7,8]
        p = minibatch_ls[rank]
        #print(f'mini batch start={p * dia_len_n * batch_size * sample_size_n}')
        #print(f'mini batch end={(p+1) * dia_len_n * batch_size * sample_size_n}')

        if model_self_absorption == True:
            P_minibatch = tc.from_numpy(P_handle['P_array'][:,:, p * dia_len_n * batch_size * sample_size_n: (p+1) * dia_len_n * batch_size * sample_size_n]).to(dev)
            n_det = P_minibatch.shape[0] 
        else:
            P_minibatch = 0
            n_det = 0
        #print(f'P_minibatch={P_minibatch}')
        model = PPM(dev, model_self_absorption, lac, X, p, n_element, n_lines, mass_attenuation_cross_section_FL,
                        detected_fl_unit_concentration, n_line_group_each_element,
                        sample_height_n, batch_size, sample_size_n, sample_size_cm,
                        probe_energy, incident_probe_intensity, model_probe_attenuation, probe_attCS_ls,
                        theta, signal_attenuation_factor,
                        n_det, P_minibatch, det_size_cm, det_from_sample_cm, det_solid_angle_ratio)
        
        y1_hat, y2_hat = model() #y1_hat dimension: (n_lines, batch_size); y2_hat dimension: (batch_size,)
        xrf_data = np.clip(y1_hat.detach().cpu().numpy(), 0, np.inf)
        xrt_data = np.exp(- y2_hat.detach().cpu().numpy())
        #### Use mpi to write the generated dataset to the hdf5 file
        with h5py.File(sim_XRF_file, 'r+') as d:
            d["exchange/data"][:, batch_size * p // sample_size_n: batch_size * (p + 1) // sample_size_n, :] = \
            np.reshape(xrf_data, (n_lines, batch_size // sample_size_n, -1)) 
            #print(d["exchange/data"].shape)
            ## shape of d["exchange/data"] = (n_lines, sample_height_n, sample_size_n)
        
        with h5py.File(sim_XRT_file, 'r+') as d:
            d["exchange/data"][3, batch_size * p // sample_size_n: batch_size * (p + 1) // sample_size_n, :] = \
            np.reshape(xrt_data, (batch_size // sample_size_n, -1))
        
            ## shape of d["exchange/data"] = (4, sample_height_n, sample_size_n)
        ####
        iteration_time = datetime.datetime.now() - start_time  # Calculate time taken for the iteration
        print(f"Batch {m + 1}/{n_batch} time cost: {iteration_time}")
    
    total_time = datetime.datetime.now() - start_time_total  # Calculate time taken for the whole simulation
    print(f"Total forward simulation time cost: {total_time}")
    with h5py.File(sim_XRF_file, 'r+') as d:
        d["exchange/data"][2, 0] = d["exchange/data"][1, 0] * d["exchange/data"][3, 0]

    del lac
    tc.cuda.empty_cache()

    ## It's important to close the hdf5 file hadle in the end of the reconstruction.
    P_handle.close()       
