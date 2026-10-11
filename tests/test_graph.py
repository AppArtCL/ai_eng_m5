"""Pruebas del grafo ReAct, la memoria y la limpieza del historial (sin API)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState

from agente.graph import AgenteState, construir_grafo, quitar_thinking_previo
from conftest import llamada

THINKING = {"type": "thinking", "thinking": "", "signature": "firma"}


def test_estado_hereda_de_messages_state() -> None:
    # MessagesState es un TypedDict: no admite issubclass, así que se revisa la base declarada.
    assert MessagesState in AgenteState.__orig_bases__
    assert "messages" in AgenteState.__annotations__


def test_grafo_tiene_el_ciclo_react() -> None:
    aristas = {(a.source, a.target) for a in construir_grafo().compile().get_graph().edges}
    assert ("__start__", "modelo") in aristas
    assert ("modelo", "herramientas") in aristas
    assert ("herramientas", "modelo") in aristas  # el ciclo
    assert ("modelo", "__end__") in aristas


def test_quitar_thinking_limpia_turnos_anteriores_y_conserva_el_actual() -> None:
    viejo = AIMessage(content=[THINKING, {"type": "text", "text": "respuesta vieja"}])
    actual = AIMessage(content=[THINKING, {"type": "text", "text": "pensando"}])
    mensajes = [HumanMessage("a"), viejo, HumanMessage("b"), actual]

    limpios = quitar_thinking_previo(mensajes)

    assert limpios[1].content == [{"type": "text", "text": "respuesta vieja"}]
    assert limpios[3].content[0]["type"] == "thinking"
    assert viejo.content[0]["type"] == "thinking"  # el original no se modifica
    assert len(limpios) == len(mensajes)


async def test_multi_paso_encadena_dos_herramientas(agente_falso) -> None:
    app = agente_falso([
        llamada("buscar_cliente_por_nombre", {"nombre": "María López"}, "t1"),
        llamada("buscar_pedidos", {"cliente_id": 310}, "t2"),
        AIMessage(content="María López tiene 6 pedidos por $31.400."),
    ])
    config = {"configurable": {"thread_id": "multi-paso"}}

    estado = await app.ainvoke({"messages": [HumanMessage("¿Pedidos de María López?")]}, config)

    resultados = [m.content for m in estado["messages"] if isinstance(m, ToolMessage)]
    assert resultados == [
        "Cliente encontrado: María López con cliente_id=310",
        "pedidos=6, total=$31400, ultimo_pedido=2026-10-03",
    ]


async def test_ciclo_de_retorno_el_error_vuelve_al_modelo(agente_falso) -> None:
    app = agente_falso([
        llamada("buscar_cliente_por_nombre", {"nombre": "Roberto Sánchez"}, "t1"),
        AIMessage(content="No encontré a Roberto Sánchez. ¿Puedes verificar el nombre?"),
    ])
    config = {"configurable": {"thread_id": "ciclo"}}

    estado = await app.ainvoke({"messages": [HumanMessage("¿Pedidos de Roberto Sánchez?")]}, config)

    tool_msg = next(m for m in estado["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content.startswith("ERROR:")
    assert isinstance(estado["messages"][-1], AIMessage)  # el grafo no se cayó


async def test_memoria_por_thread_id(agente_falso) -> None:
    app = agente_falso([AIMessage(content="uno"), AIMessage(content="dos"), AIMessage(content="tres")])

    await app.ainvoke({"messages": [HumanMessage("primera")]}, {"configurable": {"thread_id": "A"}})
    estado_a = await app.ainvoke({"messages": [HumanMessage("segunda")]}, {"configurable": {"thread_id": "A"}})
    estado_b = await app.ainvoke({"messages": [HumanMessage("otra")]}, {"configurable": {"thread_id": "B"}})

    assert len(estado_a["messages"]) == 4  # el thread A recuerda el turno anterior
    assert len(estado_b["messages"]) == 2  # el thread B parte de cero
