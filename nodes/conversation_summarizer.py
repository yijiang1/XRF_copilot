from nodeology.node import Node
from .utils import record_messages

conversation_summarizer = Node(
    node_type="summarizer",
    prompt_template="""# Instructions:
Summarize the previous conversation and output a summary of key points in bullet points.
Each bullet point should be a complete sentence and contain only one key point.
Do not add new information. Do not make up information. Do not change the order of information.
For numbers, use the exact values from the conversation. Do not make up numbers.
Output MUST be bullet points ONLY, do not add explanation before or after.""",
    sink="conversation_summary",
    use_conversation=True,
)

def conversation_summarizer_pre_process(state, client, **kwargs):
    record_messages(
        state, [("assistant", f"{state['agent_nickname']}: I will summarize our conversation.", "green")]
    )
    return state

def conversation_summarizer_post_process(state, client, **kwargs):
    if state["verbose"]:
        record_messages(state, [("assistant", f"{state['conversation_summary']}", "blue")])
    state["conversation"] = []
    return state

conversation_summarizer.pre_process = conversation_summarizer_pre_process
conversation_summarizer.post_process = conversation_summarizer_post_process 