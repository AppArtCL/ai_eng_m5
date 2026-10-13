# Pre-entrega 5: Agente de razonamiento cíclico con memoria persistente

Este es un agente que responde preguntas sobre clientes y sus pedidos. Está construido con **LangGraph** (`StateGraph` + `MessagesState`), usa **Claude Haiku 5.5** con herramientas vinculadas mediante `llm.bind_tools()` y guarda la memoria en SQLite con el checkpointer **`SqliteSaver`**, en su versión asíncrona `AsyncSqliteSaver`.

- **Razona en varios pasos.** Para saber cuántos pedidos tiene "María López", primero busca su ID y recién después consulta sus pedidos: dos llamadas a herramientas para llegar a una respuesta.
- **Recuerda la conversación, incluso después de cerrar la conexión.** La memoria vive en `checkpoints.sqlite`. Una conexión nueva recupera el historial del mismo `thread_id` y la conversación continúa.
- **No inventa datos.** Si una herramienta devuelve un error, el agente lo explica y pide una aclaración.

**Dónde está todo:** el grafo y el estado en [`src/agente/graph.py`](src/agente/graph.py), las herramientas en [`src/agente/tools.py`](src/agente/tools.py), la persistencia y los escenarios en [`src/agente/main.py`](src/agente/main.py) y la traza ReAct en [`traces/traza_ejecucion.log`](traces/traza_ejecucion.log) (texto) y [`traces/traza_ejecucion.json`](traces/traza_ejecucion.json) (JSON).

---

## Evidencia por criterio de evaluación

### 1. Arquitectura del Grafo y Gestión de Estado (30%)

**Definición explícita del estado con `MessagesState`** ([`graph.py`](src/agente/graph.py), `AgentState`). `MessagesState` define la clave `messages: Annotated[list[AnyMessage], add_messages]`. El reducer `add_messages` cumple el rol de `operator.add`: cada nodo devuelve solo sus mensajes nuevos y el reducer los **anexa** al historial en vez de sobrescribirlo, así el diálogo se mantiene coherente de un turno a otro.

```python
from langgraph.graph import MessagesState, StateGraph, START, END

# MessagesState ya incluye la lista de mensajes con el reducer add_messages
class AgentState(MessagesState):
    """Estado del grafo: hereda de MessagesState la clave `messages`. ..."""
```

**Instanciación del `StateGraph` y ciclo ReAct** ([`graph.py`](src/agente/graph.py), `construir_grafo`): el nodo del agente llama al LLM con las herramientas vinculadas (`llm.bind_tools(herramientas)`), el `ToolNode` las ejecuta, `tools_condition` decide si ir a las herramientas o terminar (`END`), y la arista `tools → agent` cierra el ciclo.

```python
from langgraph.prebuilt import ToolNode, tools_condition

grafo = StateGraph(AgentState)

# Nodos
grafo.add_node("agent", call_model)  # LLM con las herramientas vinculadas (bind_tools)
grafo.add_node("tools", ToolNode(tools=herramientas, handle_tool_errors=True))

# Aristas y ciclo ReAct
grafo.add_edge(START, "agent")
grafo.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
grafo.add_edge("tools", "agent")  # retorno cíclico al agente
```

**Compilación con checkpointer** ([`main.py`](src/agente/main.py), `main`): el grafo se compila con el checkpointer conectado y se invoca con un `thread_id`, así el agente mantiene la memoria de los turnos previos.

```python
# main()
async with AsyncSqliteSaver.from_conn_string(RUTA_DB) as checkpointer:
    app = construir_grafo().compile(checkpointer=checkpointer)

# ejecutar_turno()
config: RunnableConfig = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": 10,
}
async for paso in app.astream(entrada, config=config, stream_mode="updates"):
    ...
```

El grafo está armado a mano con `StateGraph` y `MessagesState`; no usa abstracciones como `create_react_agent`.

```mermaid
graph TD;
	__start__([__start__]):::first
	agent(agent)
	tools(tools)
	__end__([__end__]):::last
	__start__ --> agent;
	agent -.-> __end__;
	agent -.-> tools;
	tools --> agent;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

(El diagrama se generó desde el código con `construir_grafo().compile().get_graph().draw_mermaid()`.)

**El estado no crece sin control (evita el "Estado Sucio").** `call_model` le envía al modelo solo los últimos 20 mensajes con `trim_messages`, empezando siempre en un mensaje del usuario para no dejar resultados de herramientas huérfanos. El checkpointer conserva el historial completo; lo que se limita es lo que ve el modelo en cada llamada.

**Autonomía: no hay rutas `if/else` manuales.** Es el modelo el que decide si llama a una herramienta; `tools_condition` solo lee esa decisión (si el último mensaje trae `tool_calls`) y enruta.

### 2. Integración de Herramientas y Razonamiento Multi-paso (35%)

**Definición y binding formal de herramientas.** Dos herramientas propias decoradas con `@tool`, con docstrings y tipado estricto, que simulan consultas a una base de datos ([`tools.py`](src/agente/tools.py)). Se vinculan al LLM con `bind_tools` ([`graph.py`](src/agente/graph.py)):

```python
from langchain_core.tools import tool

@tool
def buscar_cliente_por_nombre(nombre: str) -> str:
    """Busca el ID interno (cliente_id) de un cliente a partir de su nombre completo.
    Usa esta herramienta SIEMPRE que el usuario mencione un cliente por su nombre
    y necesites averiguar su cliente_id antes de poder consultar sus pedidos. ..."""

@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Busca la cantidad de pedidos, el monto total gastado por un cliente y la fecha de su último pedido,
    dado su cliente_id NUMÉRICO (no su nombre — si solo tienes el nombre, primero
    usa buscar_cliente_por_nombre para obtener el ID). ..."""

herramientas: list[BaseTool] = [buscar_cliente_por_nombre, buscar_pedidos]
llm_with_tools = llm.bind_tools(herramientas)
```

**Razonamiento multi-paso genuino (dos llamadas sucesivas).** Las herramientas están separadas a propósito: los pedidos solo se buscan por `cliente_id` y el usuario pregunta por nombre, así que **una sola llamada no alcanza**. Ante "¿Cuántos pedidos tuvo María López y cuál fue el total?" el agente recorre el ciclo completo:

1. **Paso 1:** llama a `buscar_cliente_por_nombre` (el dato base).
2. **Paso 2:** recibe el `ToolMessage` con `cliente_id=310`, lo analiza en el nodo `agent` y decide llamar a `buscar_pedidos` **usando ese dato**.
3. **Paso 3:** con toda la información acumulada en el historial genera la respuesta final, sin `tool_calls`, y el grafo va a `END`.

Nadie le indica el orden: lo decide el modelo a partir de los docstrings y del resultado de cada paso.

**Evidencia: la traza de mensajes del escenario 1** ([`traces/traza_ejecucion.log`](traces/traza_ejecucion.log), ejecución real con Claude). Se ve la secuencia `AIMessage` con `tool_calls` → `ToolMessage` → `AIMessage` con nuevo `tool_calls` → `ToolMessage` → `AIMessage` final:

```text
=== [multi_paso] thread_id=memoria-c596cdd1
    llamadas a herramientas en este turno: 2
[usuario] HumanMessage ¿Cuántos pedidos tuvo María López y cuál fue el total?
[agent] AIMessage (tool_calls) → buscar_cliente_por_nombre({'nombre': 'María López'})
[tools] ToolMessage ← buscar_cliente_por_nombre: Cliente encontrado: María López con cliente_id=310
[agent] AIMessage (tool_calls) → buscar_pedidos({'cliente_id': 310})
[tools] ToolMessage ← buscar_pedidos: pedidos=6, total=$31400, ultimo_pedido=2026-10-03
[agent] AIMessage (respuesta final) María López tuvo **6 pedidos**, por un total de **$31.400 CLP**. Su último pedido fue el 3 de octubre de 2026.
```

La misma secuencia queda verificada en la prueba automatizada `test_multi_paso_encadena_dos_herramientas` ([`tests/test_graph.py`](tests/test_graph.py)): exige exactamente `HumanMessage → AIMessage → ToolMessage → AIMessage → ToolMessage → AIMessage` y comprueba que el `cliente_id` que devolvió la primera herramienta es el argumento de la segunda.

**Ciclo de retorno: cuando una herramienta falla, el agente pide aclaración.** Las herramientas *devuelven* un texto `ERROR: ...` en vez de lanzar una excepción, así el modelo lo lee y reacciona. Para errores inesperados (por ejemplo, argumentos inválidos), el `ToolNode` usa `handle_tool_errors=True`:

```text
=== [ciclo_retorno] thread_id=ciclo-retorno-7ce1413f
    llamadas a herramientas en este turno: 1
[usuario] HumanMessage ¿Cuántos pedidos tuvo Roberto Sánchez?
[agent] AIMessage (tool_calls) → buscar_cliente_por_nombre({'nombre': 'Roberto Sánchez'})
[tools] ToolMessage ← buscar_cliente_por_nombre: ERROR: No se encontró ningún cliente con el nombre 'Roberto Sánchez'. Verifica que esté escrito correctamente.
[agent] AIMessage (respuesta final) No pude encontrar a ningún cliente llamado "Roberto Sánchez" en la base de datos. ¿Podrías verificar que el nombre esté escrito correctamente? Si tienes otro nombre o un identificador de cliente, también puedo intentar buscarlo.
```

**Límite de recursión.** Cada invocación del grafo lleva `"recursion_limit": 10` ([`main.py`](src/agente/main.py), `ejecutar_turno`), para que un ciclo agent ↔ tools nunca quede corriendo sin techo.

### 3. Persistencia con SqliteSaver (20%)

La memoria se guarda en el archivo local `checkpoints.sqlite` con el checkpointer de SQLite de LangGraph (paquete `langgraph-checkpoint-sqlite`). El código cubre los tres elementos: inicialización del checkpointer, compilación del grafo con él y continuidad de la conversación por `thread_id`.

**1. Inicialización de `SqliteSaver`** ([`main.py`](src/agente/main.py), `main`). Uso **`AsyncSqliteSaver`**, la versión asíncrona de `SqliteSaver`, con el context manager recomendado por LangGraph apuntando a un archivo local. Es la versión que corresponde porque todo el grafo es asíncrono (`ainvoke` y `astream` con `asyncio`, como pide el enunciado), y el `SqliteSaver` síncrono no soporta esos métodos. Ambas versiones escriben las mismas tablas en el archivo, así que lo guardado se puede leer con cualquiera de las dos (ver el punto 3).

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # versión async de SqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    ...
```

**2. Compilación del grafo con el checkpointer.** El objeto `checkpointer` se pasa explícitamente al compilar:

```python
app = construir_grafo().compile(checkpointer=checkpointer)
```

**3. Demostración de continuidad mediante `thread_id`.** Cada invocación lleva `{"configurable": {"thread_id": thread_id}}`, y la demo lo prueba en tres niveles:

- **Dos turnos sucesivos en el mismo hilo.** El turno 1 pregunta por María López; el turno 2, con el mismo `thread_id`, pregunta solo "¿Y Juan Pérez?" y el agente lo entiende por el contexto guardado.
- **Reinicio de la sesión.** Se cierra la conexión a SQLite, se abre una nueva, se recompila el grafo y se continúa el mismo `thread_id`. El agente recupera el historial y responde una pregunta que depende de los dos turnos anteriores **sin llamar a ninguna herramienta**:

```python
# Sesión 1: escenarios 1 a 3
async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    app = construir_grafo().compile(checkpointer=checkpointer)
    ...
# La conexión ya se cerró: lo único que queda es el archivo .sqlite.

# Sesión 2: conexión nueva, mismo thread_id
async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    app = construir_grafo().compile(checkpointer=checkpointer)
    estado_recuperado = await app.aget_state(config)   # mensajes recuperados del archivo
```

- **Verificación directa del archivo en disco.** Al cerrar la sesión 1, el programa comprueba que `checkpoints.sqlite` existe y cuenta los checkpoints guardados del hilo. Al final relee el último checkpoint con el **`SqliteSaver` síncrono** (`verificar_con_sqlite_saver`), lo que demuestra que el estado quedó en disco en el formato estándar de LangGraph.

Así quedó en la traza real ([`traces/traza_ejecucion.log`](traces/traza_ejecucion.log)):

```text
=== [persistencia_sqlite] thread_id=memoria-c596cdd1
    (conexión nueva a SQLite: se recuperaron 12 mensajes del thread)
    llamadas a herramientas en este turno: 0
[usuario] HumanMessage De los dos clientes que te pregunté, ¿cuál gastó más en total?
[agent] AIMessage (respuesta final) María López gastó más en total: **$31.400 CLP** frente a los **$7.500 CLP** de Juan Pérez. La diferencia es de **$23.900 CLP**.

=== [evidencia_persistencia] thread_id=memoria-c596cdd1
    Al cerrar la sesión 1: checkpoints.sqlite existe en disco (1404.0 KB) con 14 checkpoints del thread.
    Releído al final con SqliteSaver: el último checkpoint tiene 14 mensajes.
    Primera pregunta guardada: ¿Cuántos pedidos tuvo María López y cuál fue el total?
    Última respuesta guardada: María López gastó más en total: **$31.400 CLP** frente a los **$7.500 CLP** de Juan Pérez. La diferencia es de **$23.900 CLP**.
```

(El tamaño corresponde al archivo completo, que acumula las ejecuciones anteriores; cada ejecución usa hilos nuevos. El archivo `checkpoints.sqlite` no se sube al repo porque está en `.gitignore`: se crea al ejecutar el agente.)

Además, dos pruebas automatizadas lo comprueban sobre un archivo SQLite real, sin usar la API: `test_persistencia_sobrevive_a_cerrar_la_conexion_sqlite` ([`tests/test_graph.py`](tests/test_graph.py)) y `test_sqlite_saver_lee_lo_que_guardo_async_sqlite_saver` ([`tests/test_main.py`](tests/test_main.py)).

### 4. Calidad del Repositorio y Entorno Profesional (15%)

- **Python 3.12** fijado en `.python-version` y en `requires-python`.
- **Type hints** en todas las funciones y **`asyncio`** en todo el flujo (`async def`, `ainvoke`, `astream`, `AsyncSqliteSaver`).
- **Dependencias declaradas** en `pyproject.toml`, con las versiones exactas fijadas en `uv.lock`. Se instala con `uv` o con `venv` + `pip` (ver abajo).
- **Credenciales fuera del repo:** la API key se lee desde `.env`, que está en `.gitignore`. El repo solo incluye `.env.example`.
- **Código separado por responsabilidad:** herramientas, grafo y ejecución en módulos distintos.
- **20 pruebas automatizadas con pytest** que corren sin API key.

---

## Cómo levantarlo

Necesitas Python 3.12 y una API key de Anthropic. Elige una de estas dos opciones.

**Opción A: con `venv` y `pip`**

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -e .

cp .env.example .env               # abre .env y pega tu ANTHROPIC_API_KEY
python -m agente
```

**Opción B: con [uv](https://docs.astral.sh/uv/)**

```bash
uv sync
cp .env.example .env               # abre .env y pega tu ANTHROPIC_API_KEY
uv run agente
```

El comando corre los cuatro escenarios, muestra las respuestas en la consola y deja la traza en `traces/traza_ejecucion.log` y `traces/traza_ejecucion.json`. Cada ejecución usa `thread_id` nuevos, así que puedes correrlo las veces que quieras sin que se mezcle con las anteriores.

### ¿No tienes una API key a mano?

No hay problema: la traza de una ejecución real ya viene en el repo ([`traces/`](traces/)), y las pruebas automatizadas corren **sin credenciales**. Esta es la salida de consola de esa misma ejecución:

```text
[multi_paso] ¿Cuántos pedidos tuvo María López y cuál fue el total?
→ María López tuvo **6 pedidos**, por un total de **$31.400 CLP**. Su último pedido fue el 3 de octubre de 2026.

[memoria] ¿Y Juan Pérez?
→ Juan Pérez tuvo **2 pedidos**, por un total de **$7.500 CLP**. Su último pedido fue el 22 de julio de 2026.

[ciclo_retorno] ¿Cuántos pedidos tuvo Roberto Sánchez?
→ No pude encontrar a ningún cliente llamado "Roberto Sánchez" en la base de datos. ¿Podrías verificar que el nombre esté escrito correctamente? Si tienes otro nombre o un identificador de cliente, también puedo intentar buscarlo.

(checkpoints.sqlite existe en disco: 1404.0 KB, 14 checkpoints guardados para memoria-c596cdd1)

(Conexión nueva a SQLite: 12 mensajes recuperados de memoria-c596cdd1)

[persistencia_sqlite] De los dos clientes que te pregunté, ¿cuál gastó más en total?
→ María López gastó más en total: **$31.400 CLP** frente a los **$7.500 CLP** de Juan Pérez. La diferencia es de **$23.900 CLP**.
(Verificación con SqliteSaver: el último checkpoint de memoria-c596cdd1 tiene 14 mensajes)

Traza guardada en traces/traza_ejecucion.json
Log guardado en traces/traza_ejecucion.log
```

Como el modelo genera texto, la redacción puede variar un poco entre ejecuciones, pero los datos (6 pedidos, $31.400, etc.) salen siempre de las herramientas.

---

## Escenarios de la demo

| # | Pregunta | `thread_id` | Qué demuestra |
|---|---|---|---|
| 1 | ¿Cuántos pedidos tuvo María López y cuál fue el total? | `memoria-…` | Multi-paso: dos herramientas encadenadas |
| 2 | ¿Y Juan Pérez? | el mismo que el 1 | Memoria dentro de la sesión |
| 3 | ¿Cuántos pedidos tuvo Roberto Sánchez? | `ciclo-retorno-…` (otro) | Ciclo de retorno: error de la herramienta y pedido de aclaración |
| 4 | De los dos clientes que te pregunté, ¿cuál gastó más en total? | el mismo que el 1, **con una conexión nueva a SQLite** | Persistencia: el historial se recupera del archivo |

La traza guarda **solo lo nuevo de cada turno**, sin repetir el historial: uso `astream(..., stream_mode="updates")`, que entrega lo que produjo cada nodo. El campo `nodo` de la traza muestra el recorrido por el grafo (`usuario → agent → tools → agent`).

---

## Pruebas automatizadas

```bash
uv run pytest                   # 20 pruebas, sin API key
uv run pytest -m integracion    # 1 prueba extra contra Claude real (necesita la key)
```

Con la opción A (venv), instala antes las herramientas de prueba con `pip install pytest pytest-asyncio` y luego corre `pytest`.

Para probar el grafo sin la API uso un **modelo falso** (`GenericFakeChatModel`) que devuelve respuestas guionadas. Lo único que se reemplaza es el LLM: los nodos, las aristas, el `ToolNode` y las herramientas son los reales.

| Archivo | Qué prueba |
|---|---|
| [`tests/test_tools.py`](tests/test_tools.py) | Búsqueda tolerante a tildes, mayúsculas y espacios; herramientas que devuelven `ERROR` en vez de lanzar excepciones |
| [`tests/test_graph.py`](tests/test_graph.py) | Herencia de `MessagesState`, el ciclo del grafo, el multi-paso, el ciclo de retorno, la memoria por `thread_id`, **la persistencia en SQLite al reabrir la conexión** y la limpieza del historial |
| [`tests/test_main.py`](tests/test_main.py) | La traza (solo lo nuevo de cada turno, tildes legibles), que **`SqliteSaver` relea del disco lo que guardó `AsyncSqliteSaver`** y la prueba de integración, que se salta sola si no hay key |

---

## Estructura del proyecto

```
src/agente/
├── __init__.py   # carga el .env antes que todo lo demás
├── __main__.py   # permite ejecutar con `python -m agente`
├── tools.py      # la "base de datos" simulada y las dos herramientas
├── graph.py      # el estado (AgentState hereda de MessagesState), los nodos y el StateGraph
└── main.py       # la persistencia (AsyncSqliteSaver), los escenarios y la traza
tests/            # pruebas automatizadas con pytest
traces/
├── traza_ejecucion.log    # traza ReAct legible
└── traza_ejecucion.json   # la misma traza en JSON
```

---

## Decisiones técnicas (y por qué)

**Los errores se devuelven, no se lanzan.** Si una herramienta lanzara una excepción, el grafo se caería. Al devolver un texto que empieza con `ERROR:`, el modelo lo lee como cualquier otro resultado y puede reaccionar. El system prompt refuerza esto con la instrucción de no inventar nunca datos.

**Limpieza de los bloques de razonamiento.** Claude Haiku 5.5 a veces "piensa" antes de responder, y ese razonamiento viene firmado y ligado a todo lo que había antes en la conversación. Cuando el recorte saca los mensajes más antiguos, esas firmas dejan de calzar y la API rechaza la petición. La función `quitar_thinking_previo` elimina esos bloques de los turnos anteriores y mantiene los del turno en curso, que la API necesita para continuar el ciclo de herramientas.

**El `.env` se carga en `__init__.py`.** `graph.py` crea el cliente de Anthropic apenas se importa. Cargar el `.env` en `__init__.py` asegura que la key esté disponible antes que cualquier otra cosa.

---

## Limitaciones conocidas

- **El recorte borra la memoria lejana sin avisar.** Si una conversación pasa los 20 mensajes, el modelo deja de ver el comienzo. Una mejora natural sería **resumir** los mensajes antiguos en vez de cortarlos.
- **La base de datos es un diccionario en memoria.** Sirve para demostrar el flujo; en un caso real las herramientas consultarían una base de datos de verdad.
- **El mensaje de error no sugiere nombres parecidos.** Si escribes mal un nombre, el agente pide verificarlo, pero no propone opciones similares.

---

## Stack

Python 3.12 · LangGraph · LangChain · Claude Haiku 5.5 (`langchain-anthropic`) · SQLite (`langgraph-checkpoint-sqlite` + `aiosqlite`) · pytest · uv
