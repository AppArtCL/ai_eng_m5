from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from agente.tools import herramientas

class AgentState(MessagesState):
    """Estado del grafo: hereda de MessagesState la clave `messages`.

    MessagesState define `messages: Annotated[list[AnyMessage], add_messages]`.
    El reducer `add_messages` cumple el rol de `operator.add` (agrega cada mensaje
    nuevo al historial en vez de reemplazarlo), y además evita duplicados: si
    llega un mensaje con un `id` que ya existe, lo actualiza. Así cada nodo
    devuelve solo sus mensajes nuevos y el diálogo completo se mantiene coherente.
    """

# Configuración del LLM
SYSTEM_PROMPT = SystemMessage(content="""
Eres un agente que se encarga de buscar los datos de clientes y sus compras.
Para ello cuentas con herramientas específicas que te permiten acceder a la información necesaria.
Usa siempre las herramientas, nunca inventes datos, y utiliza montos en CLP con separador de miles ($18.200)
Puedes reutilizar IDs obtenidos con herramientas antes en esta conversación.
Si una herramienta devuelve ERROR, no inventes: explica el problema y pide aclaración al usuario
""")
llm = ChatAnthropic(model_name="claude-haiku-5-5", timeout=None, stop=None)
llm_with_tools = llm.bind_tools(herramientas)

def quitar_thinking_previo(mensajes: list[BaseMessage]) -> list[BaseMessage]:
    """Quita los bloques 'thinking' de los AIMessage de turnos anteriores.
    Sus firmas quedan ligadas al historial en que se generaron, así que dejan
    de ser válidas cuando trim_messages recorta el comienzo. El turno en curso
    (desde el último HumanMessage) se conserva intacto, porque la API exige
    reenviar su thinking durante el ciclo de herramientas.
    """
    corte = max(
        (i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)),
        default=0,
    )
    limpios: list[BaseMessage] = []
    for m in mensajes[:corte]:
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            bloques = [
                b for b in m.content
                if not (isinstance(b, dict) and b.get("type") == "thinking")
            ]
            m = m.model_copy(update={"content": bloques})
        limpios.append(m)
    return limpios + mensajes[corte:]


async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Le pasa los últimos 20 mensajes al LLM y devuelve su respuesta (que puede
    incluir tool_calls o ser la respuesta final)."""
    mensajes_recortados = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=20,
        token_counter=len,
        start_on="human",
    )
    mensajes_limpios = quitar_thinking_previo(mensajes_recortados)
    respuesta = await llm_with_tools.ainvoke([SYSTEM_PROMPT, *mensajes_limpios])
    return {"messages": [respuesta]}

def construir_grafo() -> StateGraph:
    """Arma el grafo ReAct (sin compilar; el checkpointer se agrega en main.py)."""
    grafo = StateGraph(AgentState)
    # Nodos
    grafo.add_node("agent", call_model)  # LLM con las herramientas vinculadas (bind_tools)
    # handle_tool_errors: si una herramienta lanza una excepción inesperada (por ejemplo,
    # argumentos inválidos), el error vuelve al modelo como mensaje en vez de botar el grafo.
    grafo.add_node("tools", ToolNode(tools=herramientas, handle_tool_errors=True))

    # Aristas y ciclo ReAct
    grafo.add_edge(START, "agent")
    # tools_condition: si el último mensaje trae tool_calls va a "tools"; si no, a END.
    grafo.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    grafo.add_edge("tools", "agent")  # retorno cíclico al agente
    return grafo