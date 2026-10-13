import asyncio
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
# AsyncSqliteSaver es la versión asíncrona de SqliteSaver: el grafo usa ainvoke/astream,
# que el SqliteSaver síncrono no soporta. Ambos escriben las mismas tablas en el archivo.
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from agente.graph import construir_grafo

RUTA_DB = "checkpoints.sqlite"
RUTA_TRAZA = Path("traces/traza_ejecucion.json")
RUTA_LOG = Path("traces/traza_ejecucion.log")

def serializar_mensaje(m: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje en un dict apto para JSON, para la traza.

    `clase` guarda el tipo de mensaje de LangChain (HumanMessage, AIMessage,
    ToolMessage), así la traza muestra la secuencia del ciclo ReAct.
    """
    mensaje_json = {
        "clase": type(m).__name__,
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


def guardar_traza(
    turnos: list[dict[str, Any]], ruta: Path, persistencia: dict[str, Any] | None = None
) -> None:
    """Guarda la traza de todos los turnos en un archivo JSON legible.

    Si se entrega `persistencia`, se agrega la evidencia de que los checkpoints
    quedaron guardados en el archivo SQLite.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    traza: dict[str, Any] = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "turnos": turnos,
    }
    if persistencia is not None:
        traza["persistencia_sqlite"] = persistencia
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(traza, archivo, ensure_ascii=False, indent=2)
    print(f"\nTraza guardada en {ruta}")


def guardar_log(
    turnos: list[dict[str, Any]], ruta: Path, persistencia: dict[str, Any] | None = None
) -> None:
    """Escribe la traza ReAct en texto plano, un paso por línea, para leerla de corrido."""
    lineas: list[str] = []
    for turno in turnos:
        lineas.append(f"=== [{turno['escenario']}] thread_id={turno['thread_id']}")
        if "mensajes_recuperados_de_sqlite" in turno:
            lineas.append(
                f"    (conexión nueva a SQLite: se recuperaron "
                f"{turno['mensajes_recuperados_de_sqlite']} mensajes del thread)"
            )
        llamadas = sum(len(p.get("tool_calls", [])) for p in turno["pasos"])
        lineas.append(f"    llamadas a herramientas en este turno: {llamadas}")
        for paso in turno["pasos"]:
            prefijo = f"[{paso['nodo']}] {paso['clase']}"
            if paso.get("tool_calls"):
                for llamada in paso["tool_calls"]:
                    lineas.append(f"{prefijo} (tool_calls) → {llamada['nombre']}({llamada['args']})")
            elif paso["tipo"] == "tool":
                lineas.append(f"{prefijo} ← {paso['herramienta']}: {paso['contenido']}")
            elif paso["tipo"] == "ai":
                lineas.append(f"{prefijo} (respuesta final) {paso['contenido']}")
            else:
                lineas.append(f"{prefijo} {paso['contenido']}")
        lineas.append("")
    if persistencia is not None:
        disco = persistencia["disco_tras_sesion_1"]
        lectura = persistencia["verificacion"]
        lineas += [
            f"=== [evidencia_persistencia] thread_id={persistencia['thread_id']}",
            f"    Al cerrar la sesión 1: {disco['archivo']} existe en disco "
            f"({disco['tamano_kb']} KB) con {disco['checkpoints_del_thread']} checkpoints del thread.",
            f"    Releído al final con {lectura['lector']}: el último checkpoint tiene "
            f"{lectura['mensajes_en_ultimo_checkpoint']} mensajes.",
            f"    Primera pregunta guardada: {lectura['primera_pregunta']}",
            f"    Última respuesta guardada: {lectura['ultima_respuesta']}",
            "",
        ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Log guardado en {ruta}")


async def evidencia_en_disco(checkpointer: AsyncSqliteSaver, thread_id: str) -> dict[str, Any]:
    """Comprueba que el archivo SQLite existe y cuenta los checkpoints guardados del thread."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    checkpoints = [c async for c in checkpointer.alist(config)]
    archivo = Path(RUTA_DB)
    return {
        "archivo": RUTA_DB,
        "existe": archivo.exists(),
        "tamano_kb": round(archivo.stat().st_size / 1024, 1) if archivo.exists() else 0,
        "checkpoints_del_thread": len(checkpoints),
    }


def verificar_con_sqlite_saver(thread_id: str) -> dict[str, Any]:
    """Relee el último checkpoint del thread con el SqliteSaver síncrono.

    Demuestra que lo que escribió AsyncSqliteSaver quedó en disco en el formato
    estándar de LangGraph, legible por cualquier SqliteSaver.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(RUTA_DB) as lector:
        ultimo = lector.get_tuple(config)
    mensajes = ultimo.checkpoint["channel_values"].get("messages", []) if ultimo else []
    return {
        "lector": "SqliteSaver",
        "mensajes_en_ultimo_checkpoint": len(mensajes),
        "primera_pregunta": mensajes[0].text if mensajes else None,
        "ultima_respuesta": mensajes[-1].text if mensajes else None,
    }


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
        disco = await evidencia_en_disco(checkpointer, thread_memoria)
        print(
            f"\n({disco['archivo']} existe en disco: {disco['tamano_kb']} KB, "
            f"{disco['checkpoints_del_thread']} checkpoints guardados para {thread_memoria})"
        )
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

    # Verificación final: se relee el thread desde el archivo con SqliteSaver.
    lectura = await asyncio.to_thread(verificar_con_sqlite_saver, thread_memoria)
    print(
        f"(Verificación con SqliteSaver: el último checkpoint de {thread_memoria} "
        f"tiene {lectura['mensajes_en_ultimo_checkpoint']} mensajes)"
    )
    persistencia = {"thread_id": thread_memoria, "disco_tras_sesion_1": disco, "verificacion": lectura}

    guardar_traza(turnos, RUTA_TRAZA, persistencia)
    guardar_log(turnos, RUTA_LOG, persistencia)


def run() -> None:
    """Punto de entrada del comando 'agente'."""
    asyncio.run(main())
