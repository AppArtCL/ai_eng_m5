"""Pruebas del grafo ReAct, la memoria y la limpieza del historial (sin API)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState

from agente.graph import AgentState, construir_grafo, quitar_thinking_previo
from conftest import llamada

THINKING = {"type": "thinking", "thinking": "", "signature": "firma"}


def test_estado_hereda_de_messages_state() -> None:
    # MessagesState es un TypedDict: no admite issubclass, así que se revisa la base declarada.
    assert MessagesState in AgentState.__orig_bases__
    assert "messages" in AgentState.__annotations__


def test_grafo_tiene_el_ciclo_react() -> None:
    aristas = {(a.source, a.target) for a in construir_grafo().compile().get_graph().edges}
    assert ("__start__", "agent") in aristas
    assert ("agent", "tools") in aristas
    assert ("tools", "agent") in aristas  # el ciclo
    assert ("agent", "__end__") in aristas


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
    """Exige dos llamadas sucesivas: AIMessage → ToolMessage → AIMessage → ToolMessage → AIMessage."""
    app = agente_falso([
        llamada("buscar_cliente_por_nombre", {"nombre": "María López"}, "t1"),
        llamada("buscar_pedidos", {"cliente_id": 310}, "t2"),
        AIMessage(content="María López tiene 6 pedidos por $31.400."),
    ])
    config = {"configurable": {"thread_id": "multi-paso"}}

    estado = await app.ainvoke({"messages": [HumanMessage("¿Pedidos de María López?")]}, config)

    mensajes = estado["messages"]

    # La secuencia completa del ciclo ReAct: dos llamadas sucesivas antes de responder.
    assert [type(m).__name__ for m in mensajes] == [
        "HumanMessage",
        "AIMessage",   # tool_calls → buscar_cliente_por_nombre
        "ToolMessage",
        "AIMessage",   # tool_calls → buscar_pedidos (usa el dato anterior)
        "ToolMessage",
        "AIMessage",   # respuesta final, sin tool_calls → END
    ]
    assert mensajes[1].tool_calls[0]["name"] == "buscar_cliente_por_nombre"
    assert mensajes[3].tool_calls[0]["name"] == "buscar_pedidos"
    assert mensajes[5].tool_calls == []

    # El ID que devolvió la primera herramienta es el argumento de la segunda.
    assert "cliente_id=310" in mensajes[2].content
    assert mensajes[3].tool_calls[0]["args"] == {"cliente_id": 310}
    assert mensajes[4].content == "pedidos=6, total=$31400, ultimo_pedido=2026-10-03"


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


async def test_persistencia_sobrevive_a_cerrar_la_conexion_sqlite(tmp_path, monkeypatch) -> None:
    """El estado queda en el archivo .sqlite: una conexión nueva recupera el thread."""
    import agente.graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    monkeypatch.setattr(
        agente.graph, "llm_with_tools",
        GenericFakeChatModel(messages=iter([AIMessage("primera"), AIMessage("segunda")])),
    )
    db = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "persistente"}}

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = construir_grafo().compile(checkpointer=saver)
        await app.ainvoke({"messages": [HumanMessage("hola")]}, config)

    # Conexión cerrada. Se abre otra y se recompila el grafo desde cero.
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = construir_grafo().compile(checkpointer=saver)
        recuperado = await app.aget_state(config)
        assert [m.content for m in recuperado.values["messages"]] == ["hola", "primera"]

        estado = await app.ainvoke({"messages": [HumanMessage("¿te acuerdas?")]}, config)
        assert len(estado["messages"]) == 4  # el turno nuevo se suma al historial guardado


async def test_memoria_por_thread_id(agente_falso) -> None:
    app = agente_falso([AIMessage(content="uno"), AIMessage(content="dos"), AIMessage(content="tres")])

    await app.ainvoke({"messages": [HumanMessage("primera")]}, {"configurable": {"thread_id": "A"}})
    estado_a = await app.ainvoke({"messages": [HumanMessage("segunda")]}, {"configurable": {"thread_id": "A"}})
    estado_b = await app.ainvoke({"messages": [HumanMessage("otra")]}, {"configurable": {"thread_id": "B"}})

    assert len(estado_a["messages"]) == 4  # el thread A recuerda el turno anterior
    assert len(estado_b["messages"]) == 2  # el thread B parte de cero
