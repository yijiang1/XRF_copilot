from nodeology.node import Node, record_messages

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