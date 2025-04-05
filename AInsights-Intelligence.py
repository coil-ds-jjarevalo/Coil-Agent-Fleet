# ---------------------°
# --- Importaciones ---°
# ---------------------°
# General
import os
clear_command = 'cls' if os.name == 'nt' else 'clear'
os.system(clear_command)
from dotenv import load_dotenv
import gradio as gr
# import requests
import time
from typing import Any
from typing import Annotated, Literal
from typing_extensions import TypedDict
# from IPython.display import Image, display
# LangChain
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
# from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import HumanMessage
# Langgraph
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
# OpenAI
from langchain_openai import ChatOpenAI
# Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# Pydantic AI
from pydantic import BaseModel, Field
# Visualization
# import networkx as nx
# import matplotlib.pyplot as plt
# --------------------------°
# --- Configurar entorno ---°
# --------------------------°
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_api_key = os.getenv("GOOGLE_API_KEY")

google_project_id = "coil-398415"
bigquery_dataset = "coil_claro_col"
openai_model = "gpt-4o" # "o1-preview"
google_model = "gemini-2.0-flash"
selected_model = google_model
temperature = 1  # 0

# Validar variables de entorno necesarias
if not model_api_key:
    raise ValueError("No se encontró la API key del modelo en las variables de entorno.")

print("Entorno correctamenete configurado...")
time.sleep(3)
os.system(clear_command)# --------------------------------°
# --- Configurar Base de Datos ---°
# --------------------------------°
""" Base Chinook de prueba
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

response = requests.get(url)

if response.status_code == 200:
    # Open a local file in binary write mode
    with open("Chinook.db", "wb") as file:
        # Write the content of the response (the file) to the local file
        file.write(response.content)
    print("File downloaded and saved as Chinook.db")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")


db = SQLDatabase.from_uri("sqlite:///Chinook.db")
"""
bq_uri = f"bigquery://{google_project_id}/{bigquery_dataset}"
print(f"Connectando a BigQuery: {bq_uri}")
time.sleep(3)
os.system(clear_command)

try:
    db = SQLDatabase.from_uri(bq_uri) # ¡Aquí está el cambio clave!

    print(f"Dialect: {db.dialect}") # Debería imprimir 'bigquery'
    # time.sleep(1) # Opcional

    print("Fetching usable table names...")
    usable_tables = db.get_usable_table_names()
    print(f"Usable tables: {usable_tables}/n")
    # time.sleep(1) # Opcional

    # Opcional: Probar una consulta simple si quieres verificar la conexión
    # test_query = f"SELECT COUNT(*) FROM `{google_project_id}.{bigquery_dataset}.{usable_tables[0]}` LIMIT 1"
    # print(f"Running test query: {test_query}")
    # print(db.run(test_query))

except Exception as e:
    print(f"Error connecting to BigQuery or fetching tables: {e}")
    print("Please check your GOOGLE_PROJECT_ID, BIGQUERY_DATASET, and authentication setup (Application Default Credentials or GOOGLE_APPLICATION_CREDENTIALS).")
    exit() # Salir si no se puede conectar

# time.sleep(5)
# os.system(clear_command)
# db.run("SELECT * FROM Artist LIMIT 10;")
# Conexión a Big Query
# ---------------------------------°
# --- Definir funciones (tools) ---°
# ---------------------------------°
# Wrap a ToolNode with a fallback to handle errors and surface them to the agent
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Instacia de LLM con Gooogle
google_llm = ChatGoogleGenerativeAI(
    model=google_model,
    google_api_key=GOOGLE_API_KEY,
    temperature=temperature,
    convert_system_message_to_human=True # Ayuda con la compatibilidad de prompts
)

# Instacia de LLM con OpenAI
openai_llm = ChatOpenAI(model=openai_model)

selected_llm = google_llm
toolkit = SQLDatabaseToolkit(db=db, llm=selected_llm)
tools = toolkit.get_tools()
# 1) list_tables_tool: Fetch the available tables from the database
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
# 2) get_schema_tool: Fetch the DDL for a table
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
# print(list_tables_tool.invoke(""))
# print(get_schema_tool.invoke("Artist"))
# 3)db_query_tool: Execute the query and fetch the results OR return an error message if the query fails
@tool
def db_query_tool(query: str) -> str:
    """
    Execute a Google BigQuery Standard SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    Make sure to use correct BigQuery syntax, including quoting table names with backticks (`) like `project.dataset.table` if necessary.
    """
    result = db.run_no_throw(query)
    print(f"--- Query Result ---\n{result}\n-----------------------")
    if result is None or "Error:" in result: # A veces run_no_throw devuelve un string con "Error:"
         return f"Error: Query failed. Potential BigQuery error or syntax issue. Please rewrite your query using BigQuery Standard SQL syntax (use backticks for table names: `project.dataset.table`) and try again. Query attempted: {query}. Error details (if available): {result}"
    if not result: # Si el resultado es una cadena vacía (consulta exitosa pero sin filas)
        return "Query executed successfully but returned no results."
    return str(result) # Asegurarse de que sea string
# print(db_query_tool.invoke("SELECT * FROM Artist LIMIT 10;"))
#  Prompt an LLM to check for common mistakes in the query and later add this as a node in the workflow
query_check_system = """You are a Google BigQuery SQL expert with a strong attention to detail.
You are working with a database connected to the dataset `{dataset_id}` in project `{project_id}`.
Double check the incoming Google BigQuery Standard SQL query for common mistakes, including:
- Using incorrect table names or forgetting to qualify them with dataset (e.g., `my_table` instead of `{dataset_id}.my_table` or `{project_id}.{dataset_id}.my_table`). Use backticks (`) for quoting if needed: `project.dataset.table`.
- Using syntax specific to other SQL dialects (like SQLite, PostgreSQL, MySQL) that is invalid in BigQuery Standard SQL.
- Using NOT IN with NULL values.
- Using UNION when UNION ALL should have been used.
- Using BETWEEN for exclusive ranges.
- Data type mismatch in predicates (e.g., comparing STRING to INT64).
- Properly quoting identifiers (using backticks ` `).
- Using the correct number of arguments for BigQuery functions.
- Casting to the correct BigQuery data type (e.g., CAST(col AS INT64)).
- Using the proper columns for JOINs.

If there are any mistakes, rewrite the query using **valid Google BigQuery Standard SQL**.
If there are no mistakes, just reproduce the original query.

You will call the `db_query_tool` to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system.format(project_id=google_project_id, dataset_id=bigquery_dataset)), # Formatear el prompt
     ("placeholder", "{messages}")]
)

query_check = query_check_prompt | selected_llm.bind_tools( 
    [db_query_tool], tool_choice="db_query_tool"
)
# print(query_check.invoke({"messages": [("user", "SELECT * FROM Artist LIMIT 10;")]}))
# ---------------------------------°
# --- Definir el workflow ---------°
# ---------------------------------°
# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = selected_llm.bind_tools(
    [get_schema_tool]
)
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)

# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")

# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a Google BigQuery SQL expert with a strong attention to detail. You are interacting with tables within the dataset `{dataset_id}` in project `{project_id}`.

Given an input question, output a syntactically correct **Google BigQuery Standard SQL** query to run, then look at the results of the query and return the answer using the SubmitFinalAnswer tool.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer. The SQL query itself should be generated as plain text content in your response, NOT as a tool call argument.

When generating the query:
- Ensure the query uses **BigQuery Standard SQL** syntax.
- **Always qualify table names** with the dataset: `{dataset_id}.table_name`. If the table name contains special characters or resembles a keyword, quote it with backticks: `{dataset_id}.\`table-name\``. You generally do not need to include the project ID unless querying across projects.
- Unless the user specifies a specific number of examples they wish to obtain, **always limit your query to at most 5 results** using `LIMIT 5`.
- You can order the results (`ORDER BY`) by a relevant column to return the most interesting examples.
- Never query for all columns (`SELECT *`). Only select the specific columns needed to answer the question.
- If you receive an error after a query is executed, analyze the error, rewrite the query to fix it (paying close attention to BigQuery syntax and table names/quoting), and try again.
- If a query executes successfully but returns an empty result set, first double-check if the query was correct. If it was, inform the user that no data matching their request was found. DO NOT make up an answer.
- If you have enough information from the query results to answer the input question, invoke the `SubmitFinalAnswer` tool with the final answer.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system.format(project_id=google_project_id, dataset_id=bigquery_dataset)), # Formatear el prompt
     ("placeholder", "{messages}")]
)

query_gen = query_gen_prompt | selected_llm.bind_tools(
    [SubmitFinalAnswer] # SubmitFinalAnswer sigue siendo la herramienta para la respuesta final
)

def query_gen_node(state: State):
    message = query_gen.invoke(state)

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}


workflow.add_node("query_gen", query_gen_node)

# Add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)

# Add node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"


# Specify the edges between the nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

# Compile the workflow into a runnable
app = workflow.compile()
# ----------------------------------°
# --- Visualizar grafo de agente ---°
# ----------------------------------°
"""
# Get the graph from LangGraph
graph = app.get_graph()

# Create a new directed graph
G = nx.DiGraph()

# Add nodes and edges from the graph
for node in graph.nodes:
    G.add_node(node)
    for edge in graph.edges:
        if edge[0] == node:
            G.add_edge(edge[0], edge[1])

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, font_weight='bold',
        arrows=True, edge_color='gray')
plt.title("SQL Search Engine Graph")
plt.show()
"""
# ----------------------------------°
# --- Correr el Agente -------------°
# ----------------------------------°
"""
def run_chatbot():
    print("\n=== SQL Search Engine Chatbot ===")
    print("Type 'exit' or 'quit' to end the conversation")
    print("----------------------------------------")
    
    while True:
        input_prompt = input("\nYou: ").strip()
        
        if input_prompt.lower() in ['exit', 'quit']:
            print("\nGoodbye! Thanks for using SQL Search Engine Chatbot.")
            break
            
        if not input_prompt:
            print("Please enter a question.")
            continue
            
        try:
            messages = app.invoke(
                {"messages": [("user", input_prompt)]}
            )
            final_answer = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
            print("\nAssistant:", final_answer)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question in a different way.")

time.sleep(5)
os.system(clear_command)
if __name__ == "__main__":
    run_chatbot()
"""
# Interfaz con Gradio
""" Code base de ejemplo para Gradio
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
"""
def agent_chat_response(message: str, history: list[list[str]]):
    """
    Función para el chatbot de Gradio. Mantiene el historial.
    Nota: Este ejemplo simple invoca el agente *desde cero* con cada mensaje nuevo.
          Para mantener el estado real de LangGraph entre turnos, se necesitaría
          una gestión de estado más compleja (ej. almacenar/recuperar estados por sesión).
    """
    print(f"--- Chat Input: {message} ---")
    print(f"--- History: {history} ---") # History es [[user_msg1, bot_msg1], [user_msg2, bot_msg2], ...]

    try:
        # Invocar el agente con el mensaje actual del usuario
        config = {"recursion_limit": 50}
        final_state = app.invoke(
             {"messages": [HumanMessage(content=message)]},
             config=config
        )

        # Extraer la respuesta final
        final_answer = "No se pudo determinar la respuesta final."
        last_message = final_state.get("messages", [])[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
             for tc in last_message.tool_calls:
                 if tc.get("name") == "SubmitFinalAnswer":
                     final_answer = tc.get("args", {}).get("final_answer", final_answer)
                     break

        print(f"--- Chat Output: {final_answer} ---")
        return final_answer

    except Exception as e:
        import traceback
        print(f"\nError en Gradio invoke: {e}")
        print(traceback.format_exc())
        return f"Ocurrió un error: {str(e)}"

# Crear la interfaz de Chatbot
iface_chat = gr.ChatInterface(
    fn=agent_chat_response,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Pregúntame algo sobre la base de datos AInsights...", container=False, scale=7),
    title="🔮AInsights Intelligence",
    description="Chatea con un agente que puede consultar la base de datos AInsights.",
    examples=[
        "¿Cual es el motivo de contacto del caso 298652749 de la tabla salida_claro_col?"
    ]
) 
# -----------------------------------------------°
# --- Lanzar la interfaz de Gradio en consola ---°
# -----------------------------------------------°
if __name__ == "__main__":
    print("Lanzando interfaz AInsights Intelligence...")
    iface_chat.launch(share=False)