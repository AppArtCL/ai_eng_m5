"""Pruebas de la traza y una prueba opcional contra la API real."""
import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agente.main import ejecutar_turno, guardar_traza, serializar_mensaje
from conftest import llamada


def test_serializar_ai_con_thinking_y_tool_calls() -> None:
    mensaje = AIMessage(
        content=[{"type": "thinking", "thinking": "", "signature": "x"}],
        tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 310}, "id": "t1"}],
    )
    assert serializar_mensaje(mensaje) == {
        "clase": "AIMessage",
        "tipo": "ai",
        "contenido": "",
        "tool_calls": [{"nombre": "buscar_pedidos", "args": {"cliente_id": 310}}],
        "razonamiento": True,
    }


def test_serializar_tool_y_human() -> None:
    tool = ToolMessage("ok", tool_call_id="t1", name="buscar_pedidos")
    assert serializar_mensaje(tool) == {
        "clase": "ToolMessage", "tipo": "tool", "contenido": "ok", "herramienta": "buscar_pedidos",
    }
    assert serializar_mensaje(HumanMessage("hola")) == {
        "clase": "HumanMessage", "tipo": "human", "contenido": "hola",
    }


async def test_ejecutar_turno_registra_solo_lo_nuevo(agente_falso) -> None:
    app = agente_falso([
        AIMessage(content="primera respuesta"),
        llamada("buscar_pedidos", {"cliente_id": 205}, "t1"),
        AIMessage(content="Juan Pérez tiene 2 pedidos."),
    ])
    await ejecutar_turno(app, "prueba", "primera pregunta", "thread-1")

    turno = await ejecutar_turno(app, "prueba", "¿Y Juan Pérez?", "thread-1")

    assert [p["nodo"] for p in turno["pasos"]] == ["usuario", "agent", "tools", "agent"]
    assert turno["respuesta_final"] == "Juan Pérez tiene 2 pedidos."
    assert "primera pregunta" not in json.dumps(turno, ensure_ascii=False)


def test_guardar_traza_crea_carpeta_y_respeta_tildes(tmp_path: Path) -> None:
    ruta = tmp_path / "traces" / "traza.json"

    guardar_traza([{"pregunta": "¿Cuántos pedidos tuvo María López?"}], ruta)

    texto = ruta.read_text(encoding="utf-8")
    assert "María López" in texto  # sin escapes í
    assert json.loads(texto)["turnos"][0]["pregunta"] == "¿Cuántos pedidos tuvo María López?"


async def test_sqlite_saver_lee_lo_que_guardo_async_sqlite_saver(tmp_path: Path, monkeypatch) -> None:
    """Lo escrito por AsyncSqliteSaver queda en disco y SqliteSaver lo relee."""
    import agente.graph
    import agente.main
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from agente.graph import construir_grafo
    from agente.main import evidencia_en_disco, verificar_con_sqlite_saver

    db = str(tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(agente.main, "RUTA_DB", db)
    monkeypatch.setattr(
        agente.graph, "llm_with_tools", GenericFakeChatModel(messages=iter([AIMessage("guardado")]))
    )

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = construir_grafo().compile(checkpointer=saver)
        await app.ainvoke({"messages": [HumanMessage("hola")]}, {"configurable": {"thread_id": "t1"}})
        disco = await evidencia_en_disco(saver, "t1")

    assert disco["existe"] and disco["checkpoints_del_thread"] > 0

    lectura = verificar_con_sqlite_saver("t1")
    assert lectura == {
        "lector": "SqliteSaver",
        "mensajes_en_ultimo_checkpoint": 2,
        "primera_pregunta": "hola",
        "ultima_respuesta": "guardado",
    }


@pytest.mark.integracion
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="requiere ANTHROPIC_API_KEY")
async def test_integracion_agente_real_encadena_herramientas() -> None:
    """Llama a Claude de verdad. Se ejecuta solo con: uv run pytest -m integracion"""
    from langgraph.checkpoint.memory import InMemorySaver

    from agente.graph import construir_grafo

    app = construir_grafo().compile(checkpointer=InMemorySaver())
    turno = await ejecutar_turno(
        app, "integracion", "¿Cuántos pedidos tuvo María López y cuál fue el total?", "integracion-1"
    )

    herramientas = [p["herramienta"] for p in turno["pasos"] if p["tipo"] == "tool"]
    assert herramientas == ["buscar_cliente_por_nombre", "buscar_pedidos"]
    assert "31.400" in turno["respuesta_final"] or "31400" in turno["respuesta_final"]
