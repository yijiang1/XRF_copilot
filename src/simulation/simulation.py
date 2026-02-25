import os
import sys
import datetime
import numpy as np
import h5py

import xraylib as xlib
import xraylib_np as xlib_np
import torch as tc
tc.set_default_dtype(tc.float32)
import time
from .util import (
    prepare_fl_lines,
    intersecting_length_fl_detectorlet,
    ATOMIC_NUMBERS,
    rotate_3d,
    rotate_3d_inplane,
)
from .misc import print_flush_root, create_summary
from .forward_model import PPM
import warnings
import json
from tqdm import tqdm
import tifffile
warnings.filterwarnings("ignore")


def simulate_XRF_maps(params, progress_callback=None):
    """Run XRF fluorescence simulation.

    Args:
        params: JSON string or dict of simulation parameters.
        progress_callback: Optional callable(current_batch, total_batches)
            for progress reporting.

    Returns:
        (sim_XRF_file, sim_XRT_file) paths to HDF5 output files.
    """
    if isinstance(params, str):
        params_dict = json.loads(params)
    else:
        params_dict = params

    ground_truth_file = params_dict["ground_truth_file"]
    probe_energy = np.array(params_dict["probe_energy"])
    incident_probe_intensity = params_dict["incident_probe_intensity"]
    model_probe_attenuation = params_dict["model_probe_attenuation"]
    model_self_absorption = params_dict["model_self_absorption"]
    elements = params_dict["elements"]
    sample_size_cm = params_dict["sample_size_cm"]
    det_size_cm = params_dict["det_size_cm"]
    det_from_sample_cm = params_dict["det_from_sample_cm"]
    det_ds_spacing_cm = params_dict["det_ds_spacing_cm"]
    suffix = params_dict.get("suffix", "")
    debug = params_dict.get("debug", False)
    overwrite_P = False
    gpu_id = params_dict.get("gpu_id", 3)

    if tc.cuda.is_available() and gpu_id >= 0:
        dev = tc.device("cuda:{}".format(gpu_id))
    else:
        dev = "cpu"
    if debug:
        print(f"Device: {dev}")

    #### load true 3D objects ####
    X = np.load(ground_truth_file)
    if debug:
        print(f"Test objects size: {X.shape}")
    X = tc.from_numpy(X).float().to(dev)

    sample_size_n = X.shape[1]
    sample_height_n = X.shape[3]
    batch_size = sample_height_n
    if debug:
        print(f"batch_size: {batch_size}")
    dia_len_n = int(
        1.2 * (sample_height_n**2 + sample_size_n**2 + sample_size_n**2) ** 0.5
    )

    n_voxel_batch = batch_size * sample_size_n
    n_voxel = sample_height_n * sample_size_n**2

    #### parallelization ####
    n_ranks = 1
    rank = 0

    minibatch_ls_0 = tc.arange(n_ranks).to(dev)
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * batch_size)
    if debug:
        print(f'Number of batches: {n_batch}')

    #### get physical constants for the input elements and energy ####
    if debug:
        print(f'Elements: {elements}')
    n_element = len(elements)
    if debug:
        print(f'n_element: {n_element}')
    atomic_numbers = {name: ATOMIC_NUMBERS[name] for name in elements}
    if debug:
        print(f'Atomic_numbers: {atomic_numbers}')
    element_lines_roi = np.array([[element, 'K'] for element in elements])
    if debug:
        print(f'Element_lines_roi: {element_lines_roi}')

    n_line_group_each_element = np.array([np.sum(element_lines_roi[:, 0] == name) for name in elements])
    if debug:
        print(f'n_line_group_each_element: {n_line_group_each_element}')
    fl_all_lines_dic = prepare_fl_lines(
        element_lines_roi,
        n_line_group_each_element,
        probe_energy,
        sample_size_n,
        sample_size_cm,
    )
    if debug:
        print(f'fl_all_lines_dic done')

    detected_fl_unit_concentration = tc.as_tensor(fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(dev)

    mass_attenuation_cross_section_FL = tc.as_tensor(
        xlib_np.CS_Total(
            np.array(list(atomic_numbers.values())),
            fl_all_lines_dic["fl_energy"]
        )
    ).float().to(dev)

    n_line_group_each_element = tc.IntTensor(fl_all_lines_dic["n_line_group_each_element"]).to(dev)
    if debug:
        print(n_line_group_each_element)

    n_lines = fl_all_lines_dic["n_lines"]
    if debug:
        print(f'Total number of energy lines (n_lines)={n_lines}')

    channel_name_roi_ls = np.array([
        element_line_roi[0] if element_line_roi[1] == "K"
        else f"{element_line_roi[0]}_{element_line_roi[1]}"
        for element_line_roi in element_lines_roi
    ]).astype("S5")
    scaler_names = np.array(["place_holder", "us_dc", "ds_ic", "abs_ic"]).astype("S12")

    probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(np.array(list(atomic_numbers.values())), np.array([probe_energy])).flatten()).to(dev)

    #### OPTION B: estimate the solid angle by the flat surface
    det_solid_angle_ratio = (np.pi * (det_size_cm/2)**2) / (4*np.pi * det_from_sample_cm**2)

    signal_attenuation_factor = 1.0

    #### get P array ####
    output_dir = os.path.join(os.path.dirname(ground_truth_file),
                               f'det_size{det_size_cm}_spacing_{det_ds_spacing_cm}_dist{det_from_sample_cm}_sample_size{sample_size_cm}_nxy{sample_size_n}_nz{sample_height_n}')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(f'{output_dir}/P_array.h5') or overwrite_P:
        print(f'Calculating the intersecting length array P. This will take quite some time...')
        intersecting_length_fl_detectorlet(n_ranks, rank,
                                           det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                           sample_size_n, sample_size_cm, sample_height_n,
                                           output_dir, 'P_array')
        if debug:
            print(f'Completed. P is saved at {output_dir}/P_array.h5')
    else:
        if debug:
            print(f'Loading an existing intersecting length array P from {output_dir}/P_array.h5')

    P_handle = h5py.File(f'{output_dir}/P_array.h5', 'r')

    #### I/O ####
    stdout_options = {'root':0, 'output_folder': output_dir, 'save_stdout': False, 'print_terminal': True}
    timestr = str(datetime.datetime.today())
    if debug:
        print_flush_root(0, val=f"time: {timestr}", output_file='', **stdout_options)

    rotation_angles = params_dict.get('rotation_angles', [0.0, 0.0, 0.0])

    rotation_str = '_rot' + '_'.join([f"{angle:.1f}".replace('.', 'p') for angle in rotation_angles])

    suffix = params_dict.get('suffix', '')
    suffix += rotation_str

    if model_probe_attenuation:
        suffix += '_pa'
    if model_self_absorption:
        suffix += '_sa'

    if not isinstance(rotation_angles, tc.Tensor):
        rotation_angles = tc.tensor(rotation_angles, device=dev)

    sim_XRF_file = f'{output_dir}/sim_xrf_E{probe_energy}{suffix}.h5'
    sim_XRT_file = f'{output_dir}/sim_xrt_E{probe_energy}{suffix}.h5'
    params_file_name = f'sim_params_E{probe_energy}_{suffix}.txt'

    with h5py.File(sim_XRF_file, 'w') as d:
        grp = d.create_group("exchange")
        data = grp.create_dataset("data", shape=(n_lines, sample_height_n, sample_size_n), dtype="f4")
        element_names = grp.create_dataset("elements", data = channel_name_roi_ls)

    with h5py.File(sim_XRT_file, 'w') as d:
        grp = d.create_group("exchange")
        data = grp.create_dataset("data", shape=(4, sample_height_n, sample_size_n), dtype="f4")

    #### simulation ####
    start_time_total = datetime.datetime.now()

    # rotate the 3D objects
    if not tc.all(rotation_angles == 0):
        print(f"Rotating object with rotation angles (degrees): {rotation_angles.cpu().numpy()}")

        X_rot = X.clone()

        if rotation_angles[0] != 0:
            X_rot = rotate_3d_inplane(X_rot, rotation_angles[0], dev, use_degrees=True, axis='x')

        if rotation_angles[1] != 0:
            X_rot = rotate_3d_inplane(X_rot, rotation_angles[1], dev, use_degrees=True, axis='y')

        if rotation_angles[2] != 0:
            X_rot = rotate_3d_inplane(X_rot, rotation_angles[2], dev, use_degrees=True, axis='z')

        X = X_rot.clone()

    print(f'X shape: {X.shape}')
    if debug:
        for i in range(n_element):
            tifffile.imwrite(f'{output_dir}/X_{elements[i]}_rot' + '_'.join([f"{angle:.1f}".replace('.', 'p') for angle in rotation_angles])+ '.tiff', X[i].cpu().numpy())

    if model_self_absorption == True:
        lac = X.view(n_element, 1, 1, n_voxel) * mass_attenuation_cross_section_FL.view(n_element, n_lines, 1, 1)
        lac = lac.expand(-1, -1, n_voxel_batch, -1).float()
    else:
        lac = 0.

    batch_iterator = range(n_batch)
    if not debug and progress_callback is None:
        batch_iterator = tqdm(batch_iterator, desc="Processing Batches")

    for m in batch_iterator:
        # Report progress
        if progress_callback is not None:
            progress_callback(m, n_batch)

        start_time = datetime.datetime.now()
        minibatch_ls = n_ranks * m + minibatch_ls_0
        p = minibatch_ls[rank]

        if model_self_absorption == True:
            if debug:
                print(f"\nDebug info for batch {m+1}:")
                print(f"X shape: {X.shape}")
                print(f"minibatch_ls: {minibatch_ls}")
                print(f"p: {p}")

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

            if debug:
                print(f"P_minibatch shape: {P_minibatch.shape}")
                print(f"n_det: {n_det}")
        else:
            P_minibatch = 0
            n_det = 0

        try:
            if debug:
                print(f"\nModel input debug for batch {m+1}:")
                if not tc.all(rotation_angles == 0):
                    print(f"Using 3D rotation with angles: {rotation_angles.cpu().numpy()}")
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

            devices = set()
            for tensor in [X, mass_attenuation_cross_section_FL, detected_fl_unit_concentration]:
                if isinstance(tensor, tc.Tensor):
                    devices.add(tensor.device)
            if debug:
                print(f"Tensor devices: {devices}")

            if len(devices) > 1:
                raise ValueError(f"Inconsistent tensor devices found: {devices}")

            try:
                if model_self_absorption:
                    if debug:
                        print(f"Attempting with self-absorption for batch {m+1}")
                    model = PPM(dev, model_self_absorption, lac, X, p, n_element, n_lines, mass_attenuation_cross_section_FL,
                                detected_fl_unit_concentration, n_line_group_each_element,
                                sample_height_n, batch_size, sample_size_n, sample_size_cm,
                                probe_energy, incident_probe_intensity, model_probe_attenuation, probe_attCS_ls,
                                0, signal_attenuation_factor,
                                n_det, P_minibatch, det_size_cm, det_from_sample_cm, det_solid_angle_ratio)

                    tc.cuda.synchronize()
                    y1_hat, y2_hat = model()
                    tc.cuda.synchronize()
                else:
                    model = PPM(dev, model_self_absorption, lac, X, p, n_element, n_lines, mass_attenuation_cross_section_FL,
                                detected_fl_unit_concentration, n_line_group_each_element,
                                sample_height_n, batch_size, sample_size_n, sample_size_cm,
                                probe_energy, incident_probe_intensity, model_probe_attenuation, probe_attCS_ls,
                                0, signal_attenuation_factor,
                                n_det, P_minibatch, det_size_cm, det_from_sample_cm, det_solid_angle_ratio)

                    y1_hat, y2_hat = model()

            except Exception as e:
                print(f"Error with self-absorption: {str(e)}")
                tc.cuda.empty_cache()

        except Exception as e:
            if debug:
                print(f"\nError in batch {m+1}: {str(e)}")
                print("Skipping this batch and continuing...")
            try:
                tc.cuda.empty_cache()
            except:
                pass
            continue

        xrf_data = np.clip(y1_hat.detach().cpu().numpy(), 0, np.inf)
        xrt_data = np.exp(- y2_hat.detach().cpu().numpy())

        with h5py.File(sim_XRF_file, 'r+') as d:
            d["exchange/data"][:, batch_size * p // sample_size_n: batch_size * (p + 1) // sample_size_n, :] = \
            np.reshape(xrf_data, (n_lines, batch_size // sample_size_n, -1))

        with h5py.File(sim_XRT_file, 'r+') as d:
            d["exchange/data"][3, batch_size * p // sample_size_n: batch_size * (p + 1) // sample_size_n, :] = \
            np.reshape(xrt_data, (batch_size // sample_size_n, -1))

        iteration_time = datetime.datetime.now() - start_time
        if debug:
            print(f"Batch {m + 1}/{n_batch} time cost: {iteration_time}")

    # Report final progress
    if progress_callback is not None:
        progress_callback(n_batch, n_batch)

    total_time = datetime.datetime.now() - start_time_total
    if debug:
        print(f"Total forward simulation time cost: {total_time}")

    del lac
    tc.cuda.empty_cache()

    P_handle.close()

    # Save xrf and xrt data as TIFF images
    XRF_data_handle = h5py.File(sim_XRF_file, 'r')
    xrf_data = XRF_data_handle['exchange/data'][:]
    XRF_data_handle.close()

    for i in range(n_element):
        tifffile.imwrite(f'{output_dir}/sim_xrf_E{probe_energy}{suffix}_{elements[i]}.tif', xrf_data[i])

    XRT_data_handle = h5py.File(sim_XRT_file, 'r')
    xrt_data = XRT_data_handle['exchange/data'][:]
    XRT_data_handle.close()

    tifffile.imwrite(f'{output_dir}/sim_xrt_E{probe_energy}{suffix}.tif', xrt_data[-1])

    if debug:
        print("simulation done")
    print(f"sim_XRF_file: {sim_XRF_file}")
    print(f"sim_XRT_file: {sim_XRT_file}")
    return sim_XRF_file, sim_XRT_file
