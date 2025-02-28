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
from nodeology.node import as_node
import json
from tqdm import tqdm
import tifffile

warnings.filterwarnings("ignore")

#@as_node(sink="simulation_result")
def simulate_XRF_maps(params):
    params_dict = json.loads(params)
    ground_truth_file = params_dict['ground_truth_file']
    probe_energy = np.array(params_dict['probe_energy'])
    incident_probe_intensity = params_dict['incident_probe_intensity']
    model_probe_attenuation = params_dict['model_probe_attenuation']
    model_self_absorption = params_dict['model_self_absorption']
    elements = params_dict['elements']
    sample_size_cm = params_dict['sample_size_cm']
    det_size_cm = params_dict['det_size_cm']
    det_from_sample_cm = params_dict['det_from_sample_cm']
    det_ds_spacing_cm = params_dict['det_ds_spacing_cm']
    batch_size = params_dict['batch_size']
    suffix = params_dict.get('suffix', '')  # Get suffix from params, default to empty string
    debug = params_dict.get('debug', False)  # Get debug flag from params, default to False
    # P_file = params_dict['P_file']
    # gpu_id = params_dict['gpu_id']
    overwrite_P = False
    gpu_id = 0
    
    if tc.cuda.is_available() and gpu_id >= 0:  
        dev = tc.device('cuda:{}'.format(gpu_id))
    else:  
        dev = "cpu"
    if debug:
        print(f'Device: {dev}')
    ####----------------------------------------------------------------------------------####
    #### load true 3D objects ####
    X = np.load(ground_truth_file)
    if debug:
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
    if debug:
        print(f'Number of batches: {n_batch}')
    
    ####----------------------------------------------------------------------------------####
    #### get physical constants for the input elements and energy #### 

    if debug:
        print(f'Elements: {elements}')
    n_element = len(elements)
    if debug:
        print(f'n_element: {n_element}')
    # Get atomic number for each element
    atomic_numbers = {name: element(name).atomic_number for name in elements}
    if debug:
        print(f'Atomic_numbers: {atomic_numbers}')
    element_lines_roi = np.array([[element, 'K'] for element in elements])
    if debug:
        print(f'Element_lines_roi: {element_lines_roi}')
    
    # Figure out the number of FL lines for each element
    n_line_group_each_element = np.array([np.sum(element_lines_roi[:, 0] == name) for name in elements])
    if debug:
        print(f'n_line_group_each_element: {n_line_group_each_element}')
    # Create a lookup table of the fluorescence lines of interests
    fl_all_lines_dic = prepare_fl_lines(element_lines_roi,                           
                                        n_line_group_each_element, probe_energy, 
                                        sample_size_n, sample_size_cm) #cpu
    if debug:
        print(f'fl_all_lines_dic done')

    detected_fl_unit_concentration = tc.as_tensor(fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(dev)

    # Get the mass attenuation cross section for each XRF line (3rd row in Table 5.3.1) as a list ####
    mass_attenuation_cross_section_FL = tc.as_tensor(
        xlib_np.CS_Total(
            np.array(list(atomic_numbers.values())),
            fl_all_lines_dic["fl_energy"]
        )
    ).float().to(dev)  # dev
    
    n_line_group_each_element = tc.IntTensor(fl_all_lines_dic["n_line_group_each_element"]).to(dev)
    if debug:
        print(n_line_group_each_element)

    n_lines = fl_all_lines_dic["n_lines"] #scalar
    if debug:
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
    probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(np.array(list(atomic_numbers.values())), np.array([probe_energy])).flatten()).to(dev)

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
    output_dir = os.path.join(os.path.dirname(ground_truth_file), 
                               f'det_size{det_size_cm}_spacing_{det_size_cm}_dist{det_from_sample_cm}_sample_size{sample_size_cm}_nxy{sample_size_n}_nz{sample_height_n}')
    os.makedirs(output_dir, exist_ok=True)

    #Check if the P array exists, if it doesn't exist, call the function to calculate the P array and store it as a .h5 file.
    if not os.path.isfile(f'{output_dir}/P_array.h5') or overwrite_P:
        if debug:
            print(f'Calculating the intersecting length array P. This will take quite some time...')
        intersecting_length_fl_detectorlet(n_ranks, rank,
                                           det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                           sample_size_n, sample_size_cm, sample_height_n, 
                                           output_dir, 'P_array') #has to use CPU for this step
        if debug:
            print(f'Completed. P is saved at {output_dir}/P_array.h5')
    else:
        if debug:
            print(f'Loading an existing intersecting length array P from {output_dir}/P_array.h5')

    P_handle = h5py.File(f'{output_dir}/P_array.h5', 'r')

    ####----------------------------------------------------------------------------------####
    #### I/O #### 
    #stdout_options = {'root':0, 'output_folder': base_path, 'save_stdout': True, 'print_terminal': False}
    stdout_options = {'root':0, 'output_folder': output_dir, 'save_stdout': False, 'print_terminal': True}
    timestr = str(datetime.datetime.today())     
    if debug:
        print_flush_root(0, val=f"time: {timestr}", output_file='', **stdout_options)
    
    # Determine suffix based on model options
    suffix = params_dict.get('suffix', '')
    if model_probe_attenuation:
        suffix += '_pa'
    if model_self_absorption:
        suffix += '_sa'

    # Construct output file names with the updated suffix
    sim_XRF_file = f'{output_dir}/sim_xrf_{suffix}.h5'
    sim_XRT_file = f'{output_dir}/sim_xrt_{suffix}.h5'
    params_file_name = f'sim_params_{suffix}.txt'

    # initialize h5 files for saving simulated signals
    with h5py.File(sim_XRF_file, 'w') as d:
        grp = d.create_group("exchange")
        data = grp.create_dataset("data", shape=(n_lines, sample_height_n, sample_size_n), dtype="f4")
        element_names = grp.create_dataset("elements", data = channel_name_roi_ls)

    with h5py.File(sim_XRT_file, 'w') as d:
        grp = d.create_group("exchange")
        data = grp.create_dataset("data", shape=(4, sample_height_n, sample_size_n), dtype="f4")
    
    ####----------------------------------------------------------------------------------####
    #### simulation ####
    start_time_total = datetime.datetime.now()  # Start time for the simulation
    theta = tc.tensor(0.0) #rotation angle in tomography
    ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
    if model_self_absorption == True:
        # Skip rotation when theta is 0
        if theta == 0:
            if debug:
                print("Theta is 0, skipping rotation and using X directly")
            # Reshape X directly without rotation
            X_ap_rot = X.view(n_element, -1)
        else:
            X_ap_rot = rotate(X, theta, dev) #dev
            
        lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * mass_attenuation_cross_section_FL.view(n_element, n_lines, 1, 1) #dev #Eq. 5.9
        lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev
    else:
        lac = 0.

    # Use tqdm for progress bar if not in debug mode
    batch_iterator = range(n_batch)
    if not debug:
        batch_iterator = tqdm(batch_iterator, desc="Processing Batches")

    for m in batch_iterator:
        start_time = datetime.datetime.now()  # Start time for the iteration
        minibatch_ls = n_ranks * m + minibatch_ls_0  #dev, e.g. [5,6,7,8]
        p = minibatch_ls[rank]
        #print(f'mini batch start={p * dia_len_n * batch_size * sample_size_n}')
        #print(f'mini batch end={(p+1) * dia_len_n * batch_size * sample_size_n}')

        if model_self_absorption == True:
            # Add debug information first for variables we already have
            if debug:
                print(f"\nDebug info for batch {m+1}:")
                print(f"X shape: {X.shape}")
                print(f"minibatch_ls: {minibatch_ls}")
                print(f"p: {p}")
                
            # Verify indices are within bounds
            max_index = dia_len_n * batch_size * sample_size_n
            start_idx = p * max_index
            end_idx = (p+1) * max_index
            
            if debug:
                print(f"Accessing P_array indices: [{start_idx}:{end_idx}]")
                print(f"P_array shape: {P_handle['P_array'].shape}")
            
            if start_idx >= P_handle['P_array'].shape[2] or end_idx > P_handle['P_array'].shape[2]:
                if debug:
                    print(f"WARNING: Index out of bounds!")
                raise ValueError(f"Batch {m+1}: Index out of bounds in P_array access")
                
            P_minibatch = tc.from_numpy(P_handle['P_array'][:,:, start_idx:end_idx]).to(dev)
            n_det = P_minibatch.shape[0]
            
            # Now we can print P_minibatch info
            if debug:
                print(f"P_minibatch shape: {P_minibatch.shape}")
                print(f"n_det: {n_det}")
        else:
            P_minibatch = 0
            n_det = 0
        #print(f'P_minibatch={P_minibatch}')
        try:
            # Debug the input parameters to the model
            if debug:
                print(f"\nModel input debug for batch {m+1}:")
                print(f"lac shape: {lac.shape if isinstance(lac, tc.Tensor) else 'scalar 0'}")
                print(f"X shape: {X.shape}")
                print(f"n_element: {n_element}")
                print(f"n_lines: {n_lines}")
                print(f"mass_attenuation_cross_section_FL shape: {mass_attenuation_cross_section_FL.shape}")
                print(f"detected_fl_unit_concentration shape: {detected_fl_unit_concentration.shape}")
                print(f"n_line_group_each_element: {n_line_group_each_element}")
                print(f"sample_height_n: {sample_height_n}")
                print(f"batch_size: {batch_size}")
                print(f"sample_size_n: {sample_size_n}")
                
            # Verify tensor device consistency
            devices = set()
            for tensor in [X, mass_attenuation_cross_section_FL, detected_fl_unit_concentration]:
                if isinstance(tensor, tc.Tensor):
                    devices.add(tensor.device)
            if debug:
                print(f"Tensor devices: {devices}")
            
            if len(devices) > 1:
                raise ValueError(f"Inconsistent tensor devices found: {devices}")

            # Try with self-absorption first
            try:
                if model_self_absorption:
                    if debug:
                        print(f"Attempting with self-absorption for batch {m+1}")
                    model = PPM(dev, model_self_absorption, lac, X, p, n_element, n_lines, mass_attenuation_cross_section_FL,
                                detected_fl_unit_concentration, n_line_group_each_element,
                                sample_height_n, batch_size, sample_size_n, sample_size_cm,
                                probe_energy, incident_probe_intensity, model_probe_attenuation, probe_attCS_ls,
                                theta, signal_attenuation_factor,
                                n_det, P_minibatch, det_size_cm, det_from_sample_cm, det_solid_angle_ratio)
                    
                    # Try to catch CUDA errors early
                    tc.cuda.synchronize()
                    
                    y1_hat, y2_hat = model()
                    
                    # Synchronize again to catch any errors in model execution
                    tc.cuda.synchronize()
                else:
                    # No self-absorption case
                    model = PPM(dev, model_self_absorption, lac, X, p, n_element, n_lines, mass_attenuation_cross_section_FL,
                                detected_fl_unit_concentration, n_line_group_each_element,
                                sample_height_n, batch_size, sample_size_n, sample_size_cm,
                                probe_energy, incident_probe_intensity, model_probe_attenuation, probe_attCS_ls,
                                theta, signal_attenuation_factor,
                                n_det, P_minibatch, det_size_cm, det_from_sample_cm, det_solid_angle_ratio)
                    
                    y1_hat, y2_hat = model()
                    
            except Exception as e:
                if debug:
                    print(f"Error with self-absorption: {str(e)}")
                    print(f"Falling back to no self-absorption for batch {m+1}")
                
                # Clear CUDA cache
                tc.cuda.empty_cache()
                
                # Try again without self-absorption
                model = PPM(dev, False, 0, X, p, n_element, n_lines, mass_attenuation_cross_section_FL,
                            detected_fl_unit_concentration, n_line_group_each_element,
                            sample_height_n, batch_size, sample_size_n, sample_size_cm,
                            probe_energy, incident_probe_intensity, model_probe_attenuation, probe_attCS_ls,
                            theta, signal_attenuation_factor,
                            n_det, P_minibatch, det_size_cm, det_from_sample_cm, det_solid_angle_ratio)
                
                y1_hat, y2_hat = model()
            
        except Exception as e:
            if debug:
                print(f"\nError in batch {m+1}: {str(e)}")
                print("Skipping this batch and continuing...")
            
            # Clear CUDA cache in case of error
            try:
                tc.cuda.empty_cache()
            except:
                pass
            continue
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
        if debug:
            print(f"Batch {m + 1}/{n_batch} time cost: {iteration_time}")
    
    total_time = datetime.datetime.now() - start_time_total  # Calculate time taken for the whole simulation
    if debug:
        print(f"Total forward simulation time cost: {total_time}")

    # with h5py.File(sim_XRF_file, 'r+') as d:
    #     d["exchange/data"][2, 0] = d["exchange/data"][1, 0] * d["exchange/data"][3, 0]

    del lac
    tc.cuda.empty_cache()

    ## It's important to close the hdf5 file hadle in the end of the reconstruction.
    P_handle.close()

    # save xrf and xrt data to tiff images
    XRF_data_handle = h5py.File(sim_XRF_file, 'r')
    xrf_data = XRF_data_handle['exchange/data'][:]
    XRF_data_handle.close() 

    for i in range(n_element):
        tifffile.imwrite(f'{output_dir}/sim_xrf_{suffix}_{elements[i]}.tif', xrf_data[i])

    XRT_data_handle = h5py.File(sim_XRT_file, 'r')
    xrt_data = XRT_data_handle['exchange/data'][:]
    XRT_data_handle.close() 

    tifffile.imwrite(f'{output_dir}/sim_xrt_{suffix}.tif', xrt_data[-1])

    if debug:
        print('simulation done')
    return "Simulation result"
    