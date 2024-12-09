#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yi Jiang
Simulate 2D XRF projection for 3D objects
Based on Panpan's reproject_FL_signal.py script
"""

from simulation import simulate_XRF_maps
import numpy as np

params = {
            'base_path':"./data/sim_panpan", # base path to true object and simulated data
            'ground_truth_file': "grid_concentration.npy", # str: file name of the true 3D structures. Has to be a npy file under base path and stores the 4d array (n_element, nx, ny, nz)
            'output_file_base':f'sim_sample_fov0.01', # str: base name of the simulation outputs (XRF maps and XRT image) 
            'incident_probe_intensity': 1.0E7, # float: incident probe intensity (# of photons per second) (I0 in Eq 5.7)
            'probe_energy': np.array([20.0]), # np array of float: probe energy (array seems to be required by xraylib)                                      
            'model_probe_attenuation': True, # boolean: include probe attenuation effect in forward model 
            'model_self_absorption': True, # boolean: include self absorption effect in forward model 
            'element_lines_roi': np.array([['Ca', 'K'], ['Ca', 'L'], ['Sc', 'K'], ['Sc', 'L']]), # array: array of elements and their FL lines for simulation
            'sample_size_cm': 0.01, # float: physical size of the 3D volume in cm^3                                  
            'det_size_cm': 0.9, # The estimated diameter of the sensor
            'det_from_sample_cm': 1.5, # The estimated spacing between the sample and the detector                           
            'det_ds_spacing_cm': 0.4, # Set this value to the value of det_size_cm divided by a number
            'batch_size': 256,  # int: batch size for parallel processing                                 
        }


if __name__ == "__main__": 
    simulate_XRF_maps(**params)