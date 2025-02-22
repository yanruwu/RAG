from src.memory_chat import process_question
from src.doc_load import descargar_documentos
from src.preprocessing import preprocess_pdf_directory
import chainlit as cl

# --- DESCARGA Y PREPROCESAMIENTO DE DOCUMENTOS ---
descargar_documentos(urls_file="urls.txt", download_dir="docs")
preprocess_pdf_directory(pdf_directory="docs")

# --- INICIALIZACIÓN DE LA CADENA DE PROCESAMIENTO ---
@cl.on_chat_start
async def on_chat_start():
    # Define los botones para la selección de módulo
    buttons = [
        cl.Action(name="Mecánica", payload={"value" : "Classical Mechanics"}),
        cl.Action(name="Electromagnetismo", payload={"value" : "Electromagnetism"}),
        cl.Action(name="Termodinámica", payload={"value" : "Thermodynamics"}),
        cl.Action(name="Física Cuántica", payload={"value" : "Quantum Physics"})
    ]
    # Envía un mensaje con los botones para que el usuario elija
    await cl.Message(content="Interact with this action button:", actions=buttons).send()

@cl.action_callback("Mecánica")
async def on_action(action: cl.Action):
    # Recupera el valor del botón seleccionado
    module = action.payload.get("value", "General Physics")
    # Guarda el módulo en la sesión del usuario
    cl.user_session.set("module", module)
    # Confirma la selección al usuario
    await cl.Message(content=f"Módulo seleccionado: {module}").send()


@cl.on_message
async def main(message: str):
    module = cl.user_session.get("module")
    prompt_module_text = f"Answer in the context of: {module}."
    session_id = cl.user_session.get("id")
    print(prompt_module_text)
    response = process_question(message.content + prompt_module_text, session_id=session_id)
    await cl.Message(content=response).send()

