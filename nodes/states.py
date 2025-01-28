from typing import Dict, List
from nodeology.state import State, StateBaseT

class XRF_Copilot_State(State):
    #survey
    begin_conversation: bool
    end_conversation: bool
    conversation: List[dict]
    conversation_summary: str
    questions: str
    collector_response: str
    agent_name: str
    user_name: str

    # params formatter
    data_path: str
    params_desc: str
    params: Dict[str, StateBaseT]
    quality: str
    quality_history: List[str]
    params_questions: str

    #workflow control
    debug: bool

class RecommendationState(State):
    recommendation: str
    recommender_knowledge: str
    example_recommendations: Dict[str, str]
