# AInsights Intelligence 

La primera versión de este proyecto implementa un agente conversacional inteligente capaz de interactuar con una base de datos Google BigQuery utilizando lenguaje natural. El agente está construido con LangChain y LangGraph, permitiendo flujos de trabajo complejos para la consulta de datos. Ofrece la flexibilidad de elegir entre modelos de lenguaje de OpenAI (GPT-4o) y Google (Gemini). Por el momento la interfaz de usuario es una página completo pero se espera configurar una "burbuja" de chat flotante y colapsable creada con Gradio.

## Características Principales

*   **Consulta de BigQuery en Lenguaje Natural:** Hacer preguntas sobre datos en BigQuery como si se hablara con un analista.
*   **Agente LangGraph Sofisticado:**
    *   Descubre tablas disponibles.
    *   Obtiene el esquema (estructura) de las tablas relevantes.
    *   Genera consultas SQL estándar de BigQuery optimizadas.
    *   Verifica la sintaxis de la consulta antes de ejecutarla.
    *   Ejecuta la consulta en BigQuery de forma segura.
    *   Interpreta los resultados y genera una respuesta final en lenguaje natural.
*   **Soporte Dual de LLM:** Elige dinámicamente entre los potentes modelos de Google Gemini y OpenAI GPT-4 a través de un menú desplegable en la interfaz.
*   **Interfaz de Burbuja Gradio:** Una interfaz de chat moderna y no intrusiva que flota sobre la página y se puede expandir o colapsar.
*   **Manejo de Errores:** Incluye mecanismos para gestionar errores durante la ejecución de herramientas y la interacción con la base de datos.
*   **Configuración Flexible:** Gestiona claves API y configuraciones de proyecto fácilmente mediante un archivo `.env`.

## Prerrequisitos

*   Python 3.8 o superior
*   Pip (gestor de paquetes de Python)
*   Acceso a Google Cloud Platform
*   Un proyecto de Google Cloud con:
    *   La API de BigQuery habilitada.
*   **Claves API:**
    *   Una clave API de OpenAI ([platform.openai.com](https://platform.openai.com/))
    *   Una clave API de Google AI (Gemini) ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey))

## Instalación y Configuración

1.  **Clona el Repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_DIRECTORIO_DEL_PROYECTO>
    ```

2.  **Crea y Activa un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las Dependencias:**
    Asegúrate de tener un archivo `requirements.txt` en el directorio raíz del proyecto (puedes generarlo con `pip freeze > requirements.txt` si ya tienes todo instalado en tu entorno virtual).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configura las Variables de Entorno:**
    Crea un archivo llamado `.env` en la raíz del proyecto y añade tus credenciales y configuraciones:

    ```dotenv
    OPENAI_API_KEY="tu_clave_api_de_openai"
    GOOGLE_API_KEY="tu_clave_api_de_google_ai_gemini"

    # Detalles de tu proyecto BigQuery (ya presentes en el código, pero bueno tenerlos aquí por referencia)
    # GOOGLE_PROJECT_ID="coil-398415"
    # BIGQUERY_DATASET="coil_claro_col"
    ```
    *   `OPENAI_API_KEY`: Tu clave secreta de OpenAI.
    *   `GOOGLE_API_KEY`: Tu clave secreta de Google AI Studio para Gemini.
    *   **Nota Importante:** La autenticación para *acceder a BigQuery* generalmente se maneja a través de ADC (paso de prerrequisitos) y no directamente con la `GOOGLE_API_KEY` (que es para el modelo Gemini).

## Uso

1.  **Ejecuta la Aplicación:**
    ```bash
    python your_script_name.py
    ```
    (Reemplaza `your_script_name.py` con el nombre de tu archivo principal).

2.  **Accede a la Interfaz:**
    Abre tu navegador web y ve a la dirección que se muestra en la terminal (normalmente `http://127.0.0.1:7860`).

3.  **Interactúa con la Burbuja:**
    *   Verás un botón flotante (💬) en la esquina inferior derecha. Haz clic en él para abrir la ventana de chat.
    *   Una vez abierta, puedes interactuar con el chatbot.
    *   Puedes colapsar la ventana haciendo clic en la cabecera ("🔮 AInsights Intelligence"). El botón flotante volverá a aparecer.

4.  **Selecciona el Modelo:**
    Usa el menú desplegable dentro de la ventana de chat para elegir entre "Google Gemini" y "OpenAI GPT-4" antes de enviar tu pregunta.

5.  **Haz Preguntas:**
    Escribe tus preguntas sobre la base de datos `coil_claro_col` en el campo de texto y presiona Enter o haz clic en "Enviar". Ejemplos:
    *   "¿Cuál es el motivo de contacto del caso 298652749 de la tabla salida_calor_col?"
    *   "Muéstrame 3 casos de la tabla entrada_calor_col con motivo 'Consulta Facturación'."
    *   "¿Cuántas tablas hay en este dataset?"

## Tecnologías Utilizadas

*   **Python:** Lenguaje de programación principal.
*   **LangChain & LangGraph:** Frameworks para construir aplicaciones y agentes con LLMs.
*   **Gradio:** Biblioteca para crear interfaces de usuario web rápidas para modelos de machine learning.
*   **Google BigQuery:** Almacén de datos consultado.
*   **Google Generative AI (Gemini):** Modelo de lenguaje de Google.
*   **OpenAI API (GPT-4o):** Modelo de lenguaje de OpenAI.
*   **Pydantic:** Para validación de datos (usado internamente por LangChain/Gradio).
*   **python-dotenv:** Para cargar variables de entorno desde un archivo `.env`.

## Posibles Mejoras Futuras

*   **Memoria Conversacional:** Implementar memoria a largo plazo para que el agente recuerde interacciones previas dentro de una sesión.
*   **Soporte para Consultas Multi-turno:** Permitir al usuario refinar preguntas o pedir aclaraciones.
*   **Streaming de Respuestas:** Mostrar la respuesta del LLM palabra por palabra en la interfaz para una mejor percepción de la velocidad.
*   **Visualización de Errores Mejorada:** Mostrar errores de forma más amigable en la interfaz de usuario.
*   **Contenerización (Docker):** Facilitar el despliegue.
*   **Pruebas:** Añadir pruebas unitarias e de integración.

---
