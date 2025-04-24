from workflow import XRFSim
import argparse
from anl_client import Argo_Client
from nodes import XRF_Copilot_State

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XRF Copilot"
    )
    parser.add_argument("-u", type=str, default="User", help="User name")
    parser.add_argument("-v", type=bool, default=False, help="Verbose mode")
    parser.add_argument("-llm", type=str, default="gpt4o", help="LLM model")

    args = parser.parse_args()

    #workflow = XRFSim(llm=args.llm, user=args.u, verbose=args.v, debug_mode=False)
    #workflow = XRFSim(debug_mode=True)

    #result = workflow.run(ui=True)

    workflow = XRFSim(
        state_defs=XRF_Copilot_State,
        name="XRF_sim",
        llm_name=Argo_Client(args.llm),
        vlm_name="gpt-4o",
        debug_mode=False)

    questions = """- What's the full path to the ground truth objects in npy format? (String for the full path to a npy file. Should end with .npy)
        - What elements do you want to simulate?
        - What's the beam energy? (Float or int)
        - What's the incident probe intensity? (Float or int)
        - Do you want to include self-absorption in the simulation?
        - Do you want to include probe attenuation in the simulation?
        - What is the physical size of the volume in cm? (Float or int)
        - What is the diameter of the detector in cm? (Float or int)
        - What is the spacing between the sample and the detector in cm? (Float or int)
        - What is the spacing between detector pixels in cm? (Float or int)
        """

    params_desc = """| parameter name | parameter type | description |
        | :--- | :--- | :--- |
        | ground_truth_file | string | path to the ground truth objects |
        | probe_energy | float | beam energy |
        | incident_probe_intensity | float | incident probe intensity |
        | elements | a list of strings | a list of chemical symbols in abberated form (e.g. Ca, Sc, O) |
        | model_self_absorption | boolean | true if self absorption is considered in the simulation |
        | model_probe_attenuation | boolean | true if probe attenuation is considered in the simulation |
        | sample_size_cm | float | physical size of the volume in cm |
        | det_size_cm | float | diameter of the sensor in cm |
        | det_from_sample_cm | float | spacing between the sample and the detector in cm |
        | det_ds_spacing_cm | float | spacing between detector pixels in cm |
        """

    initial_state = {
        "questions": questions, 
        "params_desc": params_desc, 
        "agent_nickname": 'Fluoro', 
        "user": 'User',
        "verbose": True,
    }

    result = workflow.run(init_values=initial_state, ui=False)
