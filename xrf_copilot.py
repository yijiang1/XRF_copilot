from workflow import XRFSim, XRF_Copilot_State 
 
workflow = XRFSim(
    llm_name="gpt-4o",
    vlm_name="gpt-4o",
    save_artifacts=True,
    state_defs=XRF_Copilot_State,
)

#workflow.to_yaml("xrf_sim.yaml")
#workflow.graph.get_graph().draw_mermaid_png(output_file_path="xrf_sim.png")

questions = """- What's the full path to the ground truth objects in npy format? (String for the full path to a npy file. Should end with .npy)
- What elements do you want to simulate? (String)
- What's the beam energy? (Float or int)
- What's the incident probe intensity? (Float or int)
- Do you want to include self-absorption in the simulation? (yes or no)
- Do you want to include probe attenuation in the simulation? (yes or no)
- What is the physical size of the volume in cm? (Float or int for the volume)
- What is the estimated diameter of the sensor in cm? (Float or int for the diameter)
- What is the estimated spacing between the sample and the detector in cm? (Float or int for the spacing)
- What is the batch size for parallel calculation? (Int for the batch size)
"""

params_desc = """| parameter name | parameter type | description |
| :--- | :--- | :--- |
| ground_truth_file | string | path to the ground truth objects |
| probe_energy | float | beam energy |
| incident_probe_intensity | float | incident probe intensity |
| elements | a list of strings | a list of strings of elements name in abberated form (chemical symbols) |
| model_self_absorption | boolean | true if self absorption is considered in the simulation |
| model_probe_attenuation | boolean | true if probe attenuation is considered in the simulation |
| sample_size_cm | float | physical size of the volume in cm |
| det_size_cm | float | estimated diameter of the sensor in cm |
| det_from_sample_cm | float | estimated spacing between the sample and the detector in cm |
| batch_size | int | batch size for parallel calculation |
"""

result = workflow.run(
    {"questions": questions, 
     "params_desc": params_desc, 
     "agent_name": "Fluoro", 
     "user_name": "Yi",
     "debug": True}
)
 
#print(result["simulation_result"])