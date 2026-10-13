"""Fixtures compartidas: un LLM falso para probar el grafo sin llamar a la API."""
from collections.abc import Callable

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

import agente.graph
from agente.graph import construir_grafo


def llamada(nombre: str, args: dict, id_: str) -> AIMessage:
    """AIMessage que pide una herramienta, como lo haría Claude."""
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": id_}])


@pytest.fixture
def agente_falso(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[AIMessage]], CompiledStateGraph]:
    """Devuelve una función que compila el grafo con respuestas del modelo guionadas.

    Se reemplaza solo el LLM: los nodos, las aristas, el ToolNode y las
    herramientas reales se ejecutan igual que en producción.
    """
    def _crear(respuestas: list[AIMessage]) -> CompiledStateGraph:
        modelo_falso = GenericFakeChatModel(messages=iter(respuestas))
        monkeypatch.setattr(agente.graph, "llm_with_tools", modelo_falso)
        return construir_grafo().compile(checkpointer=InMemorySaver())

    return _crear
