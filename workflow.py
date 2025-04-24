from nodeology.workflow import Workflow
from nodeology.node import as_node
from langgraph.graph import END
from nodes import survey, formatter, XRF_Copilot_State
from simulation import simulate_XRF_maps
from typing import Dict, Optional
from nodeology.log import logger, log_print_color
from anl_client import Argo_Client
import chainlit as cl
from chainlit import Message, AskActionMessage, run_sync, AskUserMessage

@as_node(sink="human_input")
def user_input(user):
    return input(f"{user}: ")
    # human_input = run_sync(
    #     AskUserMessage(
    #         content="Please let me know how you want to change any of the parameters :)",
    #         timeout=300,
    #     ).send()
    # )["output"]
    # return human_input

class XRFSim(Workflow):
    # def __init__(self, **kwargs) -> None:

    #     self.llm = "gpt4o"
    #     self.user = "User"
    #     self.agent_nickname = "Fluoro"
    #     self.verbose = False

    #     super().__init__(
    #         name="XRF_sim",
    #         llm_name=Argo_Client(self.llm),
    #         vlm_name="gpt-4o",
    #         save_artifacts=True,
    #         state_defs=XRF_Copilot_State,
    #         **kwargs,
    #         )

    #     questions = """- What's the full path to the ground truth objects in npy format? (String for the full path to a npy file. Should end with .npy)
    #     - What elements do you want to simulate?
    #     - What's the beam energy? (Float or int)
    #     - What's the incident probe intensity? (Float or int)
    #     - Do you want to include self-absorption in the simulation?
    #     - Do you want to include probe attenuation in the simulation?
    #     - What is the physical size of the volume in cm? (Float or int)
    #     - What is the diameter of the detector in cm? (Float or int)
    #     - What is the spacing between the sample and the detector in cm? (Float or int)
    #     - What is the spacing between detector pixels in cm? (Float or int)
    #     """

    #     params_desc = """| parameter name | parameter type | description |
    #     | :--- | :--- | :--- |
    #     | ground_truth_file | string | path to the ground truth objects |
    #     | probe_energy | float | beam energy |
    #     | incident_probe_intensity | float | incident probe intensity |
    #     | elements | a list of strings | a list of chemical symbols in abberated form (e.g. Ca, Sc, O) |
    #     | model_self_absorption | boolean | true if self absorption is considered in the simulation |
    #     | model_probe_attenuation | boolean | true if probe attenuation is considered in the simulation |
    #     | sample_size_cm | float | physical size of the volume in cm |
    #     | det_size_cm | float | diameter of the sensor in cm |
    #     | det_from_sample_cm | float | spacing between the sample and the detector in cm |
    #     | det_ds_spacing_cm | float | spacing between detector pixels in cm |
    #     """
    #     # run_sync(
    #     #     Message(
    #     #         content="hello:"
    #     #     ).send()
    #     # )
    #     self.initialize({
    #         "questions": questions, 
    #         "params_desc": params_desc, 
    #         "agent_nickname": self.agent_nickname, 
    #         "user": self.user,
    #         "verbose": self.verbose,
    #         })

    def create_workflow(self):
        #log_print_color(f"{self.agent_nickname}: Hello {self.user}, thank you for letting me assist with your XRF simulation today", "green")
        #log_print_color(f"{self.agent_nickname}: First I'd like to ask some questions about your settings.", "green")

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
        
        #self.compile(interrupt_before_phrases={"params_collector": "Please enter the text to analyze."})

    # def run(self, init_values: Optional[Dict] = None) -> Dict:
    #     """Run the workflow until completion or interruption.

    #     Executes the workflow graph, handling human input and state management.

    #     Args:
    #         init_values: Optional initial state values

    #     Returns:
    #         Dict: Final workflow state values or error state
    #     """
        
    #     # Initialize graph state
    #     graph_input = (
    #         self.graph.get_state(self.langgraph_config).values
    #         if init_values is None
    #         else init_values
    #     )
    #     error_state = None
    #     current_state = self.graph.get_state(self.langgraph_config)

    #     try:
    #         while True:
    #             # Run the graph until it needs input or reaches the end
    #             for _ in self.graph.stream(graph_input, self.langgraph_config):
    #                 current_state = self.graph.get_state(self.langgraph_config)
    #                 self.save_state()

    #                 # Check if we've reached the end
    #                 if (
    #                     current_state.next is None
    #                     or len(current_state.next) == 0
    #                     or END in current_state.next
    #                     or "END" in current_state.next
    #                 ):
    #                     return current_state.values if current_state else {}

    #             # Get human input when the graph needs it
    #             human_input = self._get_human_input()
    #             if self._should_exit(human_input):
    #                 return current_state.values if current_state else {}

    #             # Update state with human input and continue
    #             self.update_state(
    #                 human_input=human_input, as_node=current_state.next[0]
    #             )
    #             self.save_state()
    #             graph_input = None  # Reset input for next iteration

    #     except Exception as e:
    #         logger.error(f"Error during workflow execution: {str(e)}")
    #         error_state = {"error": str(e)}
    #         if self.debug_mode:
    #             raise
    #         return (
    #             error_state
    #             if error_state
    #             else current_state.values if current_state else {}
    #         )