import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from agente.graph import construir_grafo

CONFIG: RunnableConfig = {"configurable": {"thread_id": "conversacion_demo_1"}, "recursion_limit": 10}

async def main():
    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app = construir_grafo().compile(checkpointer=checkpointer)
        
        # Prueba 1
        print("Prueba 1")
        print("=" * 80)
        print("Turno 1: '¿Cuántos pedidos tuvo María López y cuál fue el total?'")
        print("=" * 80)
        resultado1 = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo María López y cuál fue el total?")]},
            config=CONFIG,
        )
        print(resultado1["messages"][-1].content)
        
        print("\n" + "=" * 80)
        print("Turno 2 (mismo thread_id): '¿Y Juan Pérez?'")
        print("=" * 80)
        resultado2 = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Y Juan Pérez?")]},
            config=CONFIG,
        )
        print(resultado2["messages"][-1].content)
        
        return resultado1, resultado2

def run() -> None:
    asyncio.run(main())
