from typing import Dict, List
from nodeology.state import State, StateBaseT

class XRF_Copilot_State(State):
    # survey
    begin_conversation: bool
    end_conversation: bool
    conversation: List[dict]
    conversation_summary: str
    questions: str
    collector_response: str

    # params formatter
    data_path: str
    params_desc: str
    params: Dict[str, StateBaseT]
    quality: str
    quality_history: List[str]
    params_questions: str
    params_updater_output: str

    # workflow control
    agent_nickname: str
    user: str
    verbose: bool
    confirm_parameters: bool
    continue_simulation: bool

    # Simulation results
    sim_XRF_file: str
    sim_XRT_file: str


class RecommendationState(State):
    recommendation: str
    recommender_knowledge: str
    example_recommendations: Dict[str, str]
