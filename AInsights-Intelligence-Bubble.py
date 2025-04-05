# ---------------------°
# --- Importaciones ---°
# ---------------------°
# General
import os
import time
from typing import List, Tuple, Any
from dotenv import load_dotenv

# Gradio
import gradio as gr

# LangChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import Annotated, Literal

# OpenAI
from langchain_openai import ChatOpenAI

# Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Pydantic
from pydantic import BaseModel, Field

# --------------------------°
# --- Configurar entorno ---°
# --------------------------°
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_api_key = os.getenv("GOOGLE_API_KEY")

google_project_id = "coil-398415"
bigquery_dataset = "coil_claro_col"
openai_model = "gpt-4o"  # "o1-preview"
google_model = "gemini-2.0-flash"
temperature = 0  # 0

# Validar variables de entorno necesarias
if not model_api_key:
    raise ValueError("No se encontró la API key del modelo en las variables de entorno.")

print("Entorno correctamente configurado...")

# --------------------------------°
# --- Configurar Base de Datos ---°
# --------------------------------°
bq_uri = f"bigquery://{google_project_id}/{bigquery_dataset}"
print(f"Conectando a BigQuery: {bq_uri}")

try:
    db = SQLDatabase.from_uri(bq_uri)

    print(f"Dialect: {db.dialect}")

    print("Fetching usable table names...")
    usable_tables = db.get_usable_table_names()
    print(f"Usable tables: {usable_tables}\n")

except Exception as e:
    print(f"Error connecting to BigQuery or fetching tables: {e}")
    print("Please check your GOOGLE_PROJECT_ID, BIGQUERY_DATASET, and authentication setup.")
    exit()

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

# ---------------------------------°
# --- Herramientas de Base de Datos -°
# ---------------------------------°

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
    if result is None or "Error:" in result:  # A veces run_no_throw devuelve un string con "Error:"
         return f"Error: Query failed. Potential BigQuery error or syntax issue. Please rewrite your query using BigQuery Standard SQL syntax (use backticks for table names: `project.dataset.table`) and try again. Query attempted: {query}. Error details (if available): {result}"
    if not result:  # Si el resultado es una cadena vacía (consulta exitosa pero sin filas)
        return "Query executed successfully but returned no results."
    return str(result)  # Asegurarse de que sea string

# ---------------------------------°
# --- Herramientas de Respuesta ----°
# ---------------------------------°

class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

# ---------------------------------°
# --- Prompts del Sistema ----------°
# ---------------------------------°

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
    [("system", query_check_system.format(project_id=google_project_id, dataset_id=bigquery_dataset)),  # Formatear el prompt
     ("placeholder", "{messages}")]
)

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
    [("system", query_gen_system.format(project_id=google_project_id, dataset_id=bigquery_dataset)),  # Formatear el prompt
     ("placeholder", "{messages}")]
)

# ---------------------------------°
# --- Configuración de Modelos -----°
# ---------------------------------°

def setup_model(model_name: str):
    """Configura y retorna el modelo seleccionado."""
    if model_name == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model=google_model,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    else:  # OpenAI
        return ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )

# ---------------------------------°
# --- Definir el workflow ---------°
# ---------------------------------°
# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

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

def query_gen_node(state: State):
    message = query_gen.invoke(state)

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] not in ["SubmitFinalAnswer", "submit_answer"]:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}

# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        for tc in last_message.tool_calls:
            if tc["name"] in ["SubmitFinalAnswer", "submit_answer"]:
                return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"

# ---------------------------------°
# --- Lógica del Agente -----------°
# ---------------------------------°

def create_agent(model_name: str):
    """Crea y retorna un agente configurado con el modelo seleccionado."""
    # Configurar el modelo seleccionado
    selected_llm = setup_model(model_name)
    
    # Configurar el toolkit con el modelo seleccionado
    toolkit = SQLDatabaseToolkit(db=db, llm=selected_llm)
    tools = toolkit.get_tools()
    
    # Obtener las herramientas necesarias
    list_tables_tool = next(tool for tool in tools if tool.name in ["sql_db_list_tables", "list_tables"])
    get_schema_tool = next(tool for tool in tools if tool.name in ["sql_db_schema", "get_schema"])
    
    # Configurar los prompts y el workflow
    query_check = query_check_prompt | selected_llm.bind_tools([db_query_tool], tool_choice=None)
    model_get_schema = selected_llm.bind_tools([get_schema_tool])
    query_gen = query_gen_prompt | selected_llm.bind_tools([SubmitFinalAnswer], tool_choice=None)
    
    # Configurar el workflow
    workflow = StateGraph(State)
    
    # Agregar nodos al workflow
    workflow.add_node("first_tool_call", first_tool_call)
    workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
    workflow.add_node("model_get_schema", lambda state: {"messages": [model_get_schema.invoke(state["messages"])]})
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("correct_query", model_check_query)
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
    
    # Agregar edges al workflow
    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")
    workflow.add_conditional_edges("query_gen", should_continue)
    workflow.add_edge("correct_query", "execute_query")
    workflow.add_edge("execute_query", "query_gen")
    
    # Compilar el workflow
    return workflow.compile()

def agent_chat_response(message: str, history: List[List[str]], model_name: str):
    """
    Función para el chatbot de Gradio. Mantiene el historial y permite seleccionar el modelo.
    """
    print(f"--- Chat Input: {message} ---")
    print(f"--- History: {history} ---")
    print(f"--- Selected Model: {model_name} ---")

    try:
        # Crear el agente con el modelo seleccionado
        app = create_agent(model_name)
        
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
                 if tc.get("name") in ["SubmitFinalAnswer", "submit_answer"]:
                     final_answer = tc.get("args", {}).get("final_answer", final_answer)
                     break

        print(f"--- Chat Output: {final_answer} ---")
        return final_answer

    except Exception as e:
        import traceback
        print(f"\nError en Gradio invoke: {e}")
        print(traceback.format_exc())
        return f"Ocurrió un error: {str(e)}"

# ---------------------------------°
# --- Estilos CSS para la Burbuja -°
# ---------------------------------°

bubble_css = """
#chat-bubble-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 350px;
    height: 0;
    opacity: 0;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    background-color: white;
    transition: height 0.3s ease, opacity 0.3s ease;
    z-index: 1000;
}

#chat-bubble-container.chat-open {
    height: 550px;
    opacity: 1;
}

#chat-bubble-header {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    color: white;
    padding: 10px 15px;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

#chat-bubble-content {
    display: flex;
    flex-direction: column;
    height: calc(100% - 40px);
}

#chatbot-output-bubble {
    flex-grow: 1;
    overflow-y: auto;
    padding: 10px;
}

#chat-input-row-bubble {
    display: flex;
    padding: 10px;
    border-top: 1px solid #eee;
}

#chat-input-bubble {
    flex-grow: 1;
    margin-right: 10px;
}

#chat-toggle-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    color: white;
    border: none;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 24px;
    z-index: 1001;
    transition: opacity 0.3s ease, transform 0.3s ease;
}

#chat-toggle-button:hover {
    transform: scale(1.05);
}

#chat-bubble-container.chat-open + #chat-toggle-button {
    opacity: 0;
    pointer-events: none;
}

footer {
    display: none !important;
}
"""

# ---------------------------------°
# --- JavaScript para la Burbuja --°
# ---------------------------------°

bubble_js = bubble_js = """
function toggleChatBubble() {
    console.log('toggleChatBubble called'); // LOG 1: ¿Se llama la función?
    const container = document.getElementById('chat-bubble-container');
    if (container) {
        console.log('Container found, toggling class'); // LOG 2: ¿Encontró el contenedor?
        container.classList.toggle('chat-open');

        if (container.classList.contains('chat-open')) {
            // Enfocar el campo de texto después de una breve pausa
            setTimeout(() => {
                const inputElement = document.querySelector('#chat-input-bubble textarea'); // Gradio usa un textarea dentro del wrapper
                if (inputElement) {
                    console.log('Focusing input'); // LOG 3: ¿Intenta enfocar?
                    inputElement.focus();
                } else {
                    console.log('Input element (#chat-input-bubble textarea) not found for focus'); // LOG 4: ¿No encontró el input?
                }
            }, 300);
        }
    } else {
        console.error('Chat bubble container (#chat-bubble-container) not found!'); // ERROR LOG
    }
}

function setupBubbleListeners() {
    console.log('setupBubbleListeners called'); // LOG 5: ¿Se llama la configuración?
    const toggleButton = document.getElementById('chat-toggle-button');
    const header = document.getElementById('chat-bubble-header');

    // Esperar un poco puede ayudar si los elementos tardan en renderizarse
    setTimeout(() => {
        const currentToggleButton = document.getElementById('chat-toggle-button');
        const currentHeader = document.getElementById('chat-bubble-header');

        if (currentToggleButton) {
            console.log('Attaching listener to toggle button'); // LOG 6: ¿Encontró el botón?
            currentToggleButton.removeEventListener('click', toggleChatBubble); // Evitar duplicados si se llama varias veces
            currentToggleButton.addEventListener('click', toggleChatBubble);
        } else {
            console.error('Toggle button (#chat-toggle-button) not found!'); // ERROR LOG
        }

        if (currentHeader) {
            console.log('Attaching listener to header'); // LOG 7: ¿Encontró la cabecera?
            currentHeader.removeEventListener('click', toggleChatBubble); // Evitar duplicados
            currentHeader.addEventListener('click', toggleChatBubble);
        } else {
            console.error('Chat bubble header (#chat-bubble-header) not found!'); // ERROR LOG
        }
    }, 100); // Pequeña espera de 100ms
}

// Ejecutar la configuración. Gradio se encarga de llamarlo en el momento adecuado con demo.load
// No necesitas DOMContentLoaded aquí porque demo.load lo maneja.
setupBubbleListeners();
"""

# ---------------------------------°
# --- Interfaz de Gradio ----------°
# ---------------------------------°

with gr.Blocks(css=bubble_css) as demo:
    # Contenedor principal de la burbuja (invisible inicialmente)
    with gr.Column(elem_id="chat-bubble-container"):
        # Cabecera
        with gr.Row(elem_id="chat-bubble-header"):
            gr.Markdown("🔮 AInsights Intelligence")
        
        # Contenido
        with gr.Column(elem_id="chat-bubble-content"):
            # Display del Chat
            chatbot = gr.Chatbot(elem_id="chatbot-output-bubble", height=400)
            
            # Fila de Input
            with gr.Row(elem_id="chat-input-row-bubble"):
                msg = gr.Textbox(
                    elem_id="chat-input-bubble",
                    placeholder="Pregúntame algo sobre la base de datos AInsights...",
                    show_label=False,
                    container=False
                )
                model_selector = gr.Dropdown(
                    choices=["Google Gemini", "OpenAI GPT-4"],
                    value="Google Gemini",
                    label="Modelo",
                    container=False
                )
                submit = gr.Button("Enviar", variant="primary")
    
    # Botón flotante para abrir/cerrar
    gr.HTML('<button id="chat-toggle-button">💬</button>')
    
    # Lógica de interacción
    def handle_message_submission(user_input, history, model_name):
        if not user_input:
            return "", history
        
        bot_response = agent_chat_response(user_input, history, model_name)
        history.append((user_input, bot_response))
        
        return "", history
    
    # Conectar eventos
    submit.click(
        handle_message_submission,
        inputs=[msg, chatbot, model_selector],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        handle_message_submission,
        inputs=[msg, chatbot, model_selector],
        outputs=[msg, chatbot]
    )
    
    # Ejecutar JavaScript al cargar
    demo.load(None, None, js=f"() => {{ {bubble_js} }}")

# ---------------------------------°
# --- Lanzar la Aplicación --------°
# ---------------------------------°

if __name__ == "__main__":
    print("Lanzando interfaz AInsights Intelligence (Burbuja)...")
    demo.launch(share=False) 