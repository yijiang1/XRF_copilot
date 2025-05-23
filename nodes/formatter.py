import json
from nodeology.node import Node
from .utils import record_messages

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
        state,
        [
            (
                "assistant",
                f"{state['agent_nickname']}: I will summarize input parameters from our conversation.",
                "green",
            )
        ],
    )
    return state

def formatter_post_process(state, client, **kwargs):
    if state["verbose"]:
        params_dict = json.loads(state["params"])
        record_messages(
            state,
            [
                (
                    "assistant",
                    f"{state['agent_nickname']}: Here are the current parameters:",
                    "blue",
                ),
                (
                    "assistant",
                    "\n".join(
                        [f"{key}: {value}" for key, value in params_dict.items()]
                    ),
                    "blue",
                ),
            ],
        )
    return state

formatter.pre_process = formatter_pre_process
formatter.post_process = formatter_post_process