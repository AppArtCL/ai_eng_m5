from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, BaseMessage
from agente.tools import herramientas

class AgenteState(MessagesState):
    pass

# Datos del LLM
SYSTEM_PROMPT = SystemMessage(content="""
Eres un agente que se encarga de buscar los datos de clientes y sus compras.
Para ello cuentas con herramientas específicas que te permiten acceder a la información necesaria.
Usa siempre las herramientas, nunca inventes datos, y utiliza montos en CLP con separador de miles ($18.200)
Si una herramienta devuelve ERROR, no inventes: explica el problema y pide aclaración al usuario
""")
llm = ChatAnthropic(model_name="claude-haiku-5-5", timeout=None, stop=None)
llm_con_tools = llm.bind_tools(herramientas)

async def nodo_modelo(state: AgenteState) -> dict[str, list[BaseMessage]]:
    """Le pasa el historial completo al LLM y devuelve su respuesta (que puede
    incluir tool_calls o ser la respuesta final)."""
    respuesta = await llm_con_tools.ainvoke([SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [respuesta]}

def construir_grafo() -> StateGraph:
    """Arma el grafo ReAct (sin compilar; el checkpointer se agrega en main.py)."""
    grafo = StateGraph(AgenteState)
    grafo.add_node("modelo", nodo_modelo)
    grafo.add_node("herramientas", ToolNode(tools=herramientas))
    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges(
        "modelo",
        tools_condition,
        {"tools": "herramientas", END: END}
    )
    grafo.add_edge("herramientas", "modelo")
    return grafo