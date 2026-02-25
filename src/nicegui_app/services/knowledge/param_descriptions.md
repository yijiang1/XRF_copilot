## XRF Simulation Parameters

| Parameter Name | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| ground_truth_file | string | Full path to the ground truth 3D objects in .npy format | (required) |
| probe_energy | float | Beam energy in keV | 10.0 |
| incident_probe_intensity | float | Incident probe intensity (photons/s) | 1.0E7 |
| elements | list of strings | Chemical symbols to simulate (e.g., ["Cu", "Ca", "Fe"]) | ["Cu", "Ca"] |
| model_self_absorption | boolean | Include self-absorption in the simulation | false |
| model_probe_attenuation | boolean | Include probe attenuation along the beam path | true |
| sample_size_cm | float | Physical size of the sample volume in cm | 0.01 |
| det_size_cm | float | Diameter of the detector in cm | 0.9 |
| det_from_sample_cm | float | Distance from sample to detector in cm | 1.5 |
| det_ds_spacing_cm | float | Spacing between detector pixels in cm | 0.4 |
| rotation_angles | list of 3 floats | 3D rotation angles [x, y, z] in degrees | [0.0, 0.0, 0.0] |
| gpu_id | int | GPU device ID (-1 for CPU) | 3 |
| debug | boolean | Enable verbose debug output | false |
| suffix | string | Optional suffix for output filenames | "" |
