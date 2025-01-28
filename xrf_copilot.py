from nodeology.workflow import Workflow
from nodeology.node import as_node
from langgraph.graph import END
from workflow import XRF_Copilot_State, survey, formatter
from simulation import simulate_XRF_maps

@as_node(sink="human_input")
def user_input(user_name):    
    return input(f"{user_name}: ")
 
class XRFSim(Workflow):
    def create_workflow(self):
        self.add_node("params_collector", survey) 
        self.add_node("user_input", user_input)
        self.add_node("params_formatter", formatter, source="conversation_summary")
        self.add_node("simulation", simulate_XRF_maps)
 
        self.add_conditional_flow(
            "params_collector",
            "end_conversation",
            then="params_formatter",
            otherwise="user_input",
        )
        self.add_flow("user_input", "params_collector")
        self.add_flow("params_formatter", "simulation")
        self.add_flow("simulation", END)
 
        self.set_entry("params_collector")
        self.compile(auto_input_nodes=False)
 
 
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
| batch_size | int | batch size for parallel calculation |
"""

data_path = "test_data"
 
result = workflow.run(
    {"questions": questions, 
     "params_desc": params_desc, 
     "agent_name": "Fluoro", 
     "user_name": "Yi",
     "debug": True}
)
 
#print(result["simulation_result"])