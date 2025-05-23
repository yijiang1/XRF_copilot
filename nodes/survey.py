from nodeology.node import Node
from .utils import record_messages

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
        record_messages(
            state,
            [
                (
                    "assistant",
                    f"{state['agent_nickname']}: Thank you for your answers!",
                    "green",
                )
            ],
        )
        state["conversation"].append(
            {"role": "assistant", "content": "Thank you for your answers!"}
        )
        state["begin_conversation"] = False
        state["end_conversation"] = True
        from .conversation_summarizer import conversation_summarizer

        return conversation_summarizer(state, client, **kwargs)

    record_messages(
        state,
        [("assistant", f"{state['agent_nickname']}: {collector_response}", "green")],
    )
    state["conversation"].append({"role": "assistant", "content": collector_response})

    return state

survey.pre_process = survey_pre_process
survey.post_process = survey_post_process