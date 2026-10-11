import asyncio
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from agente.graph import construir_grafo

RUTA_TRAZA = Path("traces/traza_ejecucion.json")

def serializar_mensaje(m: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje en un dict apto para JSON, para la traza."""
    mensaje_json = {
        "tipo": m.type,
        "contenido": m.text,
    }
    if isinstance(m, AIMessage):
        mensaje_json["tool_calls"] = [{"nombre": c["name"], "args": c["args"]} for c in m.tool_calls]
        mensaje_json["razonamiento"] = any(
            isinstance(b, dict) and b.get("type") == "thinking" for b in m.content
        )
    elif isinstance(m, ToolMessage):
        mensaje_json["herramienta"] = m.name
    return mensaje_json


async def ejecutar_turno(app: CompiledStateGraph, escenario: str, pregunta: str, thread_id: str) -> dict[str, Any]:
    """Ejecuta una pregunta en el thread indicado y devuelve su traza.
    Usa stream_mode="updates" para registrar solo lo que produjo cada nodo
    en este turno, sin repetir el historial guardado por el checkpointer.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }
    entrada = {"messages": [HumanMessage(content=pregunta)]}
    pasos: list[dict[str, Any]] = [
        {"nodo": "usuario", **serializar_mensaje(entrada["messages"][0])}
    ]

    async for paso in app.astream(entrada, config=config, stream_mode="updates"):
        for nodo, actualizacion in paso.items():
            for mensaje in actualizacion["messages"]:
                pasos.append({"nodo": nodo, **serializar_mensaje(mensaje)})

    respuesta_final = pasos[-1]["contenido"]
    print(f"\n[{escenario}] {pregunta}\n→ {respuesta_final}")
    return {
        "escenario": escenario,
        "thread_id": thread_id,
        "pregunta": pregunta,
        "pasos": pasos,
        "respuesta_final": respuesta_final,
    }


def guardar_traza(turnos: list[dict[str, Any]], ruta: Path) -> None:
    """Guarda la traza de todos los turnos en un archivo JSON legible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    traza = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "turnos": turnos,
    }
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(traza, archivo, ensure_ascii=False, indent=2)
    print(f"\nTraza guardada en {ruta}")


async def main() -> None:
    """Ejecuta los escenarios de demostración y guarda la traza."""
    thread_memoria = f"memoria-{uuid4().hex[:8]}"
    thread_ciclo = f"ciclo-retorno-{uuid4().hex[:8]}"

    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app = construir_grafo().compile(checkpointer=checkpointer)

        turnos = [
            # Escenario 1. (nombre → id → pedidos)
            await ejecutar_turno(
                app,
                "memoria",
                "¿Cuántos pedidos tuvo María López y cuál fue el total?",
                thread_memoria
            ),
            # Escenario 2. (Mismo thread)
            await ejecutar_turno(
                app,
                "memoria",
                "¿Y Juan Pérez?",
                thread_memoria
            ),
            # Escenario 3. (Cliente inexistente)
            await ejecutar_turno(
                app,
                "ciclo_retorno",
                "¿Cuántos pedidos tuvo Roberto Sánchez?",
                thread_ciclo
            ),
        ]

    guardar_traza(turnos, RUTA_TRAZA)


def run() -> None:
    """Punto de entrada del comando 'agente'."""
    asyncio.run(main())
