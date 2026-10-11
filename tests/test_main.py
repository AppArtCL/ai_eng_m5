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
        "tipo": "ai",
        "contenido": "",
        "tool_calls": [{"nombre": "buscar_pedidos", "args": {"cliente_id": 310}}],
        "razonamiento": True,
    }


def test_serializar_tool_y_human() -> None:
    tool = ToolMessage("ok", tool_call_id="t1", name="buscar_pedidos")
    assert serializar_mensaje(tool) == {"tipo": "tool", "contenido": "ok", "herramienta": "buscar_pedidos"}
    assert serializar_mensaje(HumanMessage("hola")) == {"tipo": "human", "contenido": "hola"}


async def test_ejecutar_turno_registra_solo_lo_nuevo(agente_falso) -> None:
    app = agente_falso([
        AIMessage(content="primera respuesta"),
        llamada("buscar_pedidos", {"cliente_id": 205}, "t1"),
        AIMessage(content="Juan Pérez tiene 2 pedidos."),
    ])
    await ejecutar_turno(app, "prueba", "primera pregunta", "thread-1")

    turno = await ejecutar_turno(app, "prueba", "¿Y Juan Pérez?", "thread-1")

    assert [p["nodo"] for p in turno["pasos"]] == ["usuario", "modelo", "herramientas", "modelo"]
    assert turno["respuesta_final"] == "Juan Pérez tiene 2 pedidos."
    assert "primera pregunta" not in json.dumps(turno, ensure_ascii=False)


def test_guardar_traza_crea_carpeta_y_respeta_tildes(tmp_path: Path) -> None:
    ruta = tmp_path / "traces" / "traza.json"

    guardar_traza([{"pregunta": "¿Cuántos pedidos tuvo María López?"}], ruta)

    texto = ruta.read_text(encoding="utf-8")
    assert "María López" in texto  # sin escapes í
    assert json.loads(texto)["turnos"][0]["pregunta"] == "¿Cuántos pedidos tuvo María López?"


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
