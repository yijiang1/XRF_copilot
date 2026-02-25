from simulation import simulate_XRF_maps
import json
#/home/beams/USER2IDD/xrf_tomo/XRF_torch/data/sim_panpan

params = json.dumps({
    "ground_truth_file": "/mnt/micdata3/XRF_tomography/simulation/copper_structure/copper_structure_0227.npy",
    "incident_probe_intensity": 1.0E7,
    "probe_energy": 10.0,
    "model_probe_attenuation": True,
    "model_self_absorption": False,
    "elements": ["Cu", "Ca"],
    "sample_size_cm": 0.01,
    "det_size_cm": 0.9,
    "det_from_sample_cm": 1.5,
    "det_ds_spacing_cm": 0.4,
    "rotation_angles": [0.0, 60.0, 10.0], # note: beam comes from the z (3rd) axis
    "debug": False,
})

if __name__ == "__main__": 
    # Create a loop to iterate through rotation angles from 0 to 360 degrees in steps of 30 degrees
    # for angle in range(0, 390, 30):  # Going to 390 to include 360
    #     if angle == 360:  # Skip 360 as it's equivalent to 0
    #         continue
            
    #     print(f"Simulating with rotation angle: {angle} degrees")
        
    #     # Parse the current parameters
    #     params_dict = json.loads(params)
        
    #     # Update the rotation angles - keep the first two angles fixed, change only the last one
    #     params_dict["rotation_angles"] = [0.0, 20.0, float(angle)]
        
    #     # Convert back to JSON string
    #     updated_params = json.dumps(params_dict)
        
    #     # Run the simulation with the updated parameters
    #     simulate_XRF_maps(updated_params)
        
    #     print(f"Completed simulation for angle: {angle} degrees")
    
    # # Reset to original parameters
    # print("All angle simulations completed. Running with original parameters:")

    simulate_XRF_maps(params)