from nodeology.workflow import Workflow
from nodeology.node import as_node
from langgraph.graph import END
from nodes import survey, formatter, XRF_Copilot_State
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
