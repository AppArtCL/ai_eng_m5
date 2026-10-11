"""Pruebas de las herramientas y de la búsqueda por nombre (sin API)."""
import pytest

from agente.tools import buscar_cliente_por_nombre, buscar_pedidos, normalizar_nombre


@pytest.mark.parametrize(
    "entrada",
    ["María López", "maria lopez", "MARÍA  LÓPEZ", "  maría   lópez  "],
)
def test_normalizar_nombre_ignora_tildes_mayusculas_y_espacios(entrada: str) -> None:
    assert normalizar_nombre(entrada) == "maria lopez"


def test_buscar_cliente_existente_devuelve_su_id() -> None:
    resultado = buscar_cliente_por_nombre.invoke({"nombre": "maria lopez"})
    assert resultado == "Cliente encontrado: María López con cliente_id=310"


def test_buscar_cliente_inexistente_devuelve_error_sin_lanzar() -> None:
    resultado = buscar_cliente_por_nombre.invoke({"nombre": "Roberto Sánchez"})
    assert resultado.startswith("ERROR:")
    assert "Roberto Sánchez" in resultado


def test_buscar_pedidos_existente() -> None:
    resultado = buscar_pedidos.invoke({"cliente_id": 310})
    assert resultado == "pedidos=6, total=$31400, ultimo_pedido=2026-10-03"


def test_buscar_pedidos_inexistente_devuelve_error_sin_lanzar() -> None:
    resultado = buscar_pedidos.invoke({"cliente_id": 999})
    assert resultado.startswith("ERROR:")
