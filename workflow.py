import json, h5py
from nodeology.workflow import Workflow
from nodeology.node import as_node, Node
from langgraph.graph import END
from nodes import survey, formatter
from simulation import simulate_XRF_maps
import chainlit as cl
from chainlit import Message, AskUserMessage, AskActionMessage, run_sync
import plotly.graph_objects as go


@as_node(sink=["human_input"])
def ask_parameters_input(collector_response):
    human_input = run_sync(
        AskUserMessage(
            content=collector_response,
            timeout=3000,
        ).send()
    )["output"]
    return human_input


@as_node(sink=[])
def display_parameters(params):
    run_sync(
        Message(
            content="Below are the current simulation parameters:",
            elements=[
                cl.CustomElement(
                    name="DataDisplay",
                    props={
                        "data": json.loads(params),
                        "showEmptyEntry": True,
                        "columns": 1,
                    },
                )
            ],
        ).send()
    )
    return


@as_node(sink="confirm_parameters")
def ask_confirm_parameters():
    res = run_sync(
        AskActionMessage(
            content="Are you happy with the parameters?",
            timeout=3000,
            actions=[
                cl.Action(
                    name="yes",
                    payload={"value": "yes"},
                    label="Yes",
                ),
                cl.Action(
                    name="no",
                    payload={"value": "no"},
                    label="No",
                ),
            ],
        ).send()
    )
    if res and res.get("payload").get("value") == "yes":
        return True
    else:
        return False


@as_node(sink=["human_input"])
def ask_parameters_updates():
    human_input = run_sync(
        AskUserMessage(
            content="Please let me know how you want to change any of the parameters :)",
            timeout=300,
        ).send()
    )["output"]
    return human_input


parameters_updater = Node(
    node_type="parameters_updater",
    prompt_template="""Update the parameters based on the user's input.

Current parameters:
{params}

User input:
{human_input}

Please return the updated parameters in JSON format with the exact same keys and structure.
    """,
    sink="params",
    sink_format="json",
)


@as_node(sink="continue_simulation")
def ask_continue_simulation():
    res = run_sync(
        AskActionMessage(
            content="Would you like to continue the simulation?",
            timeout=3000,
            actions=[
                cl.Action(
                    name="continue",
                    payload={"value": "continue"},
                    label="Continue Simulation",
                ),
                cl.Action(
                    name="finish",
                    payload={"value": "finish"},
                    label="Finish",
                ),
            ],
        ).send()
    )

    # Return the user's choice
    if res and res.get("payload").get("value") == "continue":
        return True
    else:
        return False


@as_node(sink=[])
def display_simulation_results(sim_XRF_file, sim_XRT_file):
    # Read XRF and XRT data from HDF5 files
    XRF_data_handle = h5py.File(sim_XRF_file, "r")
    xrf_data = XRF_data_handle["exchange/data"][:]
    elements = [el.decode("utf-8") for el in XRF_data_handle["exchange/elements"][:]]
    XRF_data_handle.close()

    XRT_data_handle = h5py.File(sim_XRT_file, "r")
    xrt_data = XRT_data_handle["exchange/data"][:]
    XRT_data_handle.close()

    # Create and send XRF maps for each element
    for i, element in enumerate(elements):
        fig = go.Figure(
            data=go.Heatmap(z=xrf_data[i], colorscale="Viridis", showscale=True)
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        run_sync(
            Message(
                content=f"XRF map for element {element}:",
                elements=[cl.Plotly(figure=fig)],
            ).send()
        )

    # Create and send XRT map
    fig = go.Figure(
        data=go.Heatmap(z=xrt_data[-1], colorscale="Viridis", showscale=True)
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    run_sync(
        Message(
            content="XRT map:",
            elements=[cl.Plotly(figure=fig)],
        ).send()
    )
    return


class XRFSim(Workflow):
    def create_workflow(self):
        self.add_node("params_collector", survey)
        self.add_node("ask_parameters_input", ask_parameters_input)
        self.add_node("params_formatter", formatter, source="conversation_summary")
        self.add_node("display_parameters", display_parameters)
        self.add_node("ask_confirm_parameters", ask_confirm_parameters)
        self.add_node("ask_parameters_updates", ask_parameters_updates)
        self.add_node("update_parameters", parameters_updater)
        self.add_node("simulation", simulate_XRF_maps)
        self.add_node("display_simulation_results", display_simulation_results)
        self.add_node("ask_continue_simulation", ask_continue_simulation)

        self.add_conditional_flow(
            "params_collector",
            "end_conversation",
            then="params_formatter",
            otherwise="ask_parameters_input",
        )
        self.add_flow("ask_parameters_input", "params_collector")
        self.add_flow("params_formatter", "display_parameters")
        self.add_flow("display_parameters", "ask_confirm_parameters")
        self.add_conditional_flow(
            "ask_confirm_parameters",
            "confirm_parameters",
            then="simulation",
            otherwise="ask_parameters_updates",
        )
        self.add_flow("ask_parameters_updates", "update_parameters")
        self.add_flow("update_parameters", "display_parameters")
        self.add_flow("simulation", "display_simulation_results")
        self.add_flow("display_simulation_results", "ask_continue_simulation")
        self.add_conditional_flow(
            "ask_continue_simulation",
            "continue_simulation",
            then="display_parameters",
            otherwise=END,
        )

        self.set_entry("params_collector")
        self.compile(auto_input_nodes=False)