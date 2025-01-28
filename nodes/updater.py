import json
from nodeology.node import Node, record_messages

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