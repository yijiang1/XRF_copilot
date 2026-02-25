# Suggestion Rules for XRF Simulation Parameters

## Probe Energy Selection
- For K-edge excitation: probe energy should be 1-3 keV above the element's K absorption edge
- For L-edge excitation: probe energy should be above the L3 absorption edge
- Common energies: 10 keV (general purpose), 12 keV (heavier elements), 7 keV (light elements)

## Element Configuration
- Always include the primary elements of interest
- Consider adding matrix elements (e.g., Si, O for geological samples)
- Order elements by atomic number for consistency

## Detector Geometry
- `det_size_cm`: Typical silicon drift detector active area ~0.5-1.0 cm
- `det_from_sample_cm`: Standard working distance 1.0-2.0 cm; closer = more solid angle but risk of shadowing
- `det_ds_spacing_cm`: Pixel spacing on detector; 0.3-0.5 cm typical

## Sample Configuration
- `sample_size_cm`: Should match actual sample dimension; typical range 0.001-0.1 cm for micro-XRF
- Larger samples require more computation time

## Physics Model
- Enable `model_probe_attenuation` for thick/dense samples (recommended by default)
- Enable `model_self_absorption` when sample contains high-Z elements or is optically thick for fluorescence energies
- Both disabled = fastest but least accurate simulation

## GPU Selection
- `gpu_id`: Set to the available GPU index; use `nvidia-smi` to check
- Default: 3 (assumes multi-GPU workstation)

## Rotation Angles
- Format: [theta_x, theta_y, theta_z] in degrees
- [0, 0, 0] = no rotation (standard geometry)
- For tomography: vary theta_y in steps (e.g., 0-180 degrees)
