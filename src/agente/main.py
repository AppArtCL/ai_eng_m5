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

RUTA_DB = "checkpoints.sqlite"
RUTA_TRAZA = Path("traces/traza_ejecucion.json")
RUTA_LOG = Path("traces/traza_ejecucion.log")

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


def guardar_log(turnos: list[dict[str, Any]], ruta: Path) -> None:
    """Escribe la traza ReAct en texto plano, un paso por línea, para leerla de corrido."""
    lineas: list[str] = []
    for turno in turnos:
        lineas.append(f"=== [{turno['escenario']}] thread_id={turno['thread_id']}")
        if "mensajes_recuperados_de_sqlite" in turno:
            lineas.append(
                f"    (conexión nueva a SQLite: se recuperaron "
                f"{turno['mensajes_recuperados_de_sqlite']} mensajes del thread)"
            )
        for paso in turno["pasos"]:
            if paso.get("tool_calls"):
                for llamada in paso["tool_calls"]:
                    lineas.append(f"[{paso['nodo']}] → llama {llamada['nombre']}({llamada['args']})")
            elif paso["tipo"] == "tool":
                lineas.append(f"[{paso['nodo']}] ← {paso['herramienta']}: {paso['contenido']}")
            else:
                lineas.append(f"[{paso['nodo']}] {paso['contenido']}")
        lineas.append("")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Log guardado en {ruta}")


async def main() -> None:
    """Ejecuta los escenarios de demostración y guarda la traza.

    Los escenarios 1 a 3 corren con una conexión a SQLite. Después esa conexión
    se cierra y el escenario 4 abre una nueva: si el agente sigue recordando la
    conversación, es porque el estado quedó guardado en checkpoints.sqlite.
    """
    thread_memoria = f"memoria-{uuid4().hex[:8]}"
    thread_ciclo = f"ciclo-retorno-{uuid4().hex[:8]}"

    # Sesión 1: primera conexión a SQLite.
    async with AsyncSqliteSaver.from_conn_string(RUTA_DB) as checkpointer:
        app = construir_grafo().compile(checkpointer=checkpointer)

        turnos = [
            # Escenario 1: multi-paso (nombre → cliente_id → pedidos)
            await ejecutar_turno(
                app,
                "multi_paso",
                "¿Cuántos pedidos tuvo María López y cuál fue el total?",
                thread_memoria,
            ),
            # Escenario 2: memoria (mismo thread_id, pregunta incompleta)
            await ejecutar_turno(
                app,
                "memoria",
                "¿Y Juan Pérez?",
                thread_memoria,
            ),
            # Escenario 3: ciclo de retorno (otro thread_id, cliente inexistente)
            await ejecutar_turno(
                app,
                "ciclo_retorno",
                "¿Cuántos pedidos tuvo Roberto Sánchez?",
                thread_ciclo,
            ),
        ]
    # Aquí la conexión ya está cerrada: lo único que queda es el archivo .sqlite.

    # Sesión 2: conexión nueva y grafo recompilado, mismo thread_id.
    async with AsyncSqliteSaver.from_conn_string(RUTA_DB) as checkpointer:
        app = construir_grafo().compile(checkpointer=checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": thread_memoria}}
        estado_recuperado = await app.aget_state(config)
        recuperados = len(estado_recuperado.values.get("messages", []))
        print(f"\n(Conexión nueva a SQLite: {recuperados} mensajes recuperados de {thread_memoria})")

        # Escenario 4: persistencia (la respuesta depende de los turnos 1 y 2)
        turno_persistencia = await ejecutar_turno(
            app,
            "persistencia_sqlite",
            "De los dos clientes que te pregunté, ¿cuál gastó más en total?",
            thread_memoria,
        )
        turno_persistencia["mensajes_recuperados_de_sqlite"] = recuperados
        turnos.append(turno_persistencia)

    guardar_traza(turnos, RUTA_TRAZA)
    guardar_log(turnos, RUTA_LOG)


def run() -> None:
    """Punto de entrada del comando 'agente'."""
    asyncio.run(main())
