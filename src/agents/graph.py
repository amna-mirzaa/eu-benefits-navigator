"""Wires the five agents into a linear LangGraph pipeline."""

from functools import lru_cache

from langgraph.graph import END, StateGraph

from src.agents.eligibility_agent import eligibility_node
from src.agents.explainer_agent import explainer_node
from src.agents.intake_agent import intake_node
from src.agents.retrieval_agent import retrieval_node
from src.agents.state import NavigatorState
from src.agents.verifier_agent import verifier_node


def build_graph():
    graph = StateGraph(NavigatorState)

    graph.add_node("intake", intake_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("assess", eligibility_node)
    graph.add_node("explain", explainer_node)
    graph.add_node("verify", verifier_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_edge("assess", "explain")
    graph.add_edge("explain", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run_navigator(user_input: str) -> NavigatorState:
    app = get_graph()
    return app.invoke({"user_input": user_input, "trace": []})


def run_navigator_stream(user_input: str):
    """Yields (node_name, state_so_far) after each agent completes.

    Lets the UI show real-time progress through the pipeline instead of a
    single opaque spinner for the whole multi-agent run.
    """
    app = get_graph()
    state: NavigatorState = {"user_input": user_input, "trace": []}
    for update in app.stream({"user_input": user_input, "trace": []}, stream_mode="updates"):
        for node_name, partial in update.items():
            state.update(partial)
            yield node_name, state
