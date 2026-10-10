import unicodedata
from langchain_core.tools import tool, BaseTool

CLIENTES_DB: dict[str, int] = {
    "ana garcía": 102,
    "juan pérez": 205,
    "maría lópez": 310,
    "carlos sánchez": 104,
    "laura martínez": 412,
    "diego fernández": 523,
    "sofía torres": 218,
    "pedro ramírez": 331,
    "valentina rojas": 409,
    "lucas castro": 115,
}

PEDIDOS_DB: dict[int, dict[str, int | str]] = {
    102: {"pedidos": 4, "total": 18200, "ultimo_pedido": "2026-09-14"},
    205: {"pedidos": 2, "total": 7500, "ultimo_pedido": "2026-07-22"},
    310: {"pedidos": 6, "total": 31400, "ultimo_pedido": "2026-10-03"},
    104: {"pedidos": 1, "total": 4200, "ultimo_pedido": "2026-06-18"},
    412: {"pedidos": 3, "total": 12900, "ultimo_pedido": "2026-08-29"},
    523: {"pedidos": 5, "total": 24100, "ultimo_pedido": "2026-10-09"},
    218: {"pedidos": 2, "total": 8600, "ultimo_pedido": "2026-07-05"},
    331: {"pedidos": 7, "total": 39500, "ultimo_pedido": "2026-09-27"},
    409: {"pedidos": 1, "total": 5100, "ultimo_pedido": "2026-06-30"},
    115: {"pedidos": 4, "total": 19800, "ultimo_pedido": "2026-08-14"},
}

def normalizar_nombre(texto: str) -> str:
    """Normaliza un nombre para comparar: pasa a minúsculas, colapsa espacios y elimina tildes."""
    texto = " ".join(texto.split()).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto

_INDICE_CLIENTES: dict[str, str] = {normalizar_nombre(k): k for k in CLIENTES_DB}

@tool
def buscar_cliente_por_nombre(nombre: str) -> str:
    """Busca el ID interno (cliente_id) de un cliente a partir de su nombre completo.
    Usa esta herramienta SIEMPRE que el usuario mencione un cliente por su nombre
    y necesites averiguar su cliente_id antes de poder consultar sus pedidos.
    Te devuelve un mensaje de ERROR si el nombre no coincide con ningún
    cliente registrado ignorando mayúsculas, tildes y espacios extra.
    """
    clave = normalizar_nombre(nombre)

    if clave not in _INDICE_CLIENTES:
        return f"ERROR: No se encontró ningún cliente con el nombre '{nombre}'. Verifica que esté escrito correctamente."

    nombre_real = _INDICE_CLIENTES[clave]
    return f"Cliente encontrado: {nombre_real.title()} con cliente_id={CLIENTES_DB[nombre_real]}"


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Busca la cantidad de pedidos, el monto total gastado por un cliente y la fecha de su último pedido,
    dado su cliente_id NUMÉRICO (no su nombre — si solo tienes el nombre, primero
    usa buscar_cliente_por_nombre para obtener el ID).
    Devuelve un mensaje de ERROR si el cliente_id no existe en la base de datos."""
    if cliente_id not in PEDIDOS_DB:
        return f"ERROR: No se encontraron pedidos para el cliente_id={cliente_id}"
    datos = PEDIDOS_DB[cliente_id]
    return (
        f"pedidos={datos['pedidos']}, total=${datos['total']}, ultimo_pedido={datos['ultimo_pedido']}"
    )

herramientas: list[BaseTool] = [buscar_cliente_por_nombre, buscar_pedidos]