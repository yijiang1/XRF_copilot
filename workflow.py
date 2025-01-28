import json
from typing import Dict, List
from nodeology.state import State, StateBaseT
from nodeology.node import Node, record_messages

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

    #workflow contro
    debug: bool


conversation_summarizer = Node(
    node_type="summarizer",
    prompt_template="""# Instructions:
Summarize the previous conversation and output a summary of ONLY user responses in bullet points.
Each bullet point should be a complete sentence and contain only one key point.
Do not add new information. Do not make up information. Do not change the order of information.
For numbers, use the exact values from the conversation. Do not make up numbers.
Output MUST be bullet points ONLY, do not add explanation before or after.""",
    sink="conversation_summary",
    use_conversation=True,
)


def conversation_summarizer_pre_process(state, client, **kwargs):
    record_messages(
        state, [(state["agent_name"], "I will summarize the previous conversation.", "green")]
    )
    return state


def conversation_summarizer_post_process(state, client, **kwargs):
    if state["debug"]:
        record_messages(state, [(state["agent_name"], state["conversation_summary"], "blue")])
    state["conversation"] = []
    return state


conversation_summarizer.pre_process = conversation_summarizer_pre_process
conversation_summarizer.post_process = conversation_summarizer_post_process


survey = Node(
    node_type="survey",
    prompt_template="""# QUESTIONS:
{questions}

# Instructions:
Ask ALL questions from pre-defined QUESTIONS one by one.
Ask ONLY ONE question at a time following the pre-defined order.
YOU NEED TO ASK ALL QUESTIONS! DO NOT SKIP QUESTIONS! DO NOT CHANGE ORDER OF QUESTIONS! 
DO NOT REWRITE QUESTIONS! DO NOT SHOW THE CONTENT IN THE PARENTHESIS!
Check if the user response satisfy the expected format defined in the parenthesis. If not, ask the question again.
If all questions have been asked, output exactly "COLLECT_COMPLETE".""",
    sink="collector_response",
    use_conversation=True,
)


def survey_pre_process(state, client, **kwargs):
    if "questions" not in state and "questions" not in kwargs:
        raise ValueError(f"Questions state not found")

    if len(state.get("conversation", [])) == 0:
        state["conversation"] = []
        state["begin_conversation"] = True
        state["end_conversation"] = False
        return state

    state["conversation"].append({"role": "user", "content": state["human_input"]})
    return state

def survey_post_process(state, client, **kwargs):
    collector_response = state["collector_response"]

    if "COLLECT_COMPLETE" in collector_response:
        record_messages(state, [(state["agent_name"], "Thank you for your answers!", "green")])
        state["conversation"].append(
            {"role": "assistant", "content": "Thank you for your answers!"}
        )
        state["begin_conversation"] = False
        state["end_conversation"] = True
        return conversation_summarizer(state, client, **kwargs)

    record_messages(state, [(state["agent_name"], collector_response, "green")])
    state["conversation"].append({"role": "assistant", "content": collector_response})
    return state


survey.pre_process = survey_pre_process
survey.post_process = survey_post_process

class RecommendationState(State):
    recommendation: str
    recommender_knowledge: str
    example_recommendations: Dict[str, str]


# Formatter Template
formatter = Node(
    node_type="formatter",
    prompt_template="""# PARAMS DESCRIPTION: 
{params_desc}

# SOURCE:
{source}

# Instructions:
Extract parameters from SOURCE into JSON. Include ALL parameters described in PARAMS DESCRIPTION in the JSON. 
If a parameter is not mentioned in SOURCE, use the default values in the PARAMS DESCRIPTION. Do not make up values. 
Output MUST be JSON ONLY, do not add explanation before or after the JSON.""",
    sink=["params"],
    sink_format="json",
)


def formatter_pre_process(state, client, **kwargs):
    record_messages(
        state, [(state["agent_name"], f"I will summarize input parameters from our conversation.", "green")]
    )
    return state


def formatter_post_process(state, client, **kwargs):
    
    if state["debug"]:
        params_dict = json.loads(state["params"])
        record_messages(
            state,
            [
                (state["agent_name"], "Here are the current parameters:", "blue"),
            #(state["agent_name"], state["params"], "blue"),
            #(state["agent_name"], json.dumps(state["params"], indent=4), "blue"),
                (state["agent_name"], "\n".join([f"{key}: {value}" for key, value in params_dict.items()]), "blue"),
            
            ],
        )
    
    return state


formatter.pre_process = formatter_pre_process
formatter.post_process = formatter_post_process

# Recommender Template
recommender = Node(
    node_type="recommender",
    prompt_template="""# RECOMMENDER KNOWLEDGE:
{recommender_knowledge}

# SOURCE:
{source}

# Instructions:
Check SOURCE, recommend parameters according to RECOMMENDER KNOWLEDGE.""",
    sink=["recommendation"],
)


def recommender_pre_process(state, client, **kwargs):
    record_messages(
        state,
        [(state["agent_name"], f"I will recommend some parameters based on SOURCE.", "green")],
    )
    return state


def recommender_post_process(state, client, **kwargs):
    record_messages(state, [(state["agent_name"], state["recommendation"], "blue")])
    return state


recommender.pre_process = recommender_pre_process
recommender.post_process = recommender_post_process

# Updater Template
updater = Node(
    node_type="updater",
    prompt_template="""# PARAMS DESCRIPTION:
{params_desc}

# CURRENT PARAMETERS:
{params}

# SOURCE:
{source}

# Instructions:
Modify CURRENT PARAMETERS according to SOURCE.
Output an updated JSON following PARAMS DESCRIPTION. 
Output MUST be JSON ONLY, do not add explanation before or after the JSON.""",
    sink=["params"],
    sink_format="json",
)


def updater_pre_process(state, client, **kwargs):
    source_type = kwargs.get("source", "recommendation")

    if source_type == "human_input":
        if not state["begin_conversation"]:
            if state["previous_node_type"] != "updater":
                if state["previous_node_type"] != "formatter":
                    record_messages(
                        state,
                        [
                            (state["agent_name"], "Here are the current parameters:", "green"),
                            (
                                state["agent_name"],
                                json.dumps(state["params"], indent=2),
                                "blue",
                            ),
                        ],
                    )
                record_messages(
                    state,
                    [
                        (
                            state["agent_name"],
                            'Let me know if you want to make any more changes. If all looks good, please say "LGTM".',
                            "green",
                        )
                    ],
                )
            record_messages(
                state,
                [
                    (
                        state["agent_name"],
                        'You can say "terminate pear" to terminate the workflow at any time.',
                        "yellow",
                    )
                ],
            )
            state["begin_conversation"] = True
            state["end_conversation"] = False
            state["conversation"] = []
            return None  # Signal to skip LLM call

        if any(
            keyword in state["human_input"].lower()
            for keyword in ["confirm", "quit", "exit", "bye", "lgtm", "terminate"]
        ):
            record_messages(
                state, [(state["agent_name"], "Thank you for your feedback!", "green")]
            )
            state["begin_conversation"] = False
            state["end_conversation"] = True
            state["conversation"] = []
            return None  # Signal to skip LLM call
    else:
        record_messages(
            state,
            [
                (
                    state["agent_name"],
                    f"I will update the parameters according to {source_type}.",
                    "green",
                )
            ],
        )
    return state


def updater_post_process(state, client, **kwargs):
    record_messages(
        state,
        [
            (state["agent_name"], "Here are the updated parameters:", "green"),
            (state["agent_name"], json.dumps(state["params"], indent=2), "blue"),
            (
                state["agent_name"],
                'Let me know if you want to make any more changes. If all looks good, please say "LGTM".',
                "green",
            ),
        ],
    )
    return state


updater.pre_process = updater_pre_process
updater.post_process = updater_post_process
