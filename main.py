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
        cl.Action(name="Mecánica", payload={"value": "Classical Mechanics"}),
        cl.Action(name="Electromagnetismo", payload={"value": "Electromagnetism"}),
        cl.Action(name="Termodinámica", payload={"value": "Thermodynamics"}),
        cl.Action(name="Física Cuántica", payload={"value": "Quantum Physics"}),
        cl.Action(name="General", payload={"value": "General Physics"}),
    ]
    # Envía un mensaje para que el usuario elija el área
    await cl.Message(content="Hola! Estoy aquí para ayudarte con cualquier duda. Formula tu pregunta o especifica un área primero.", actions=buttons).send()

@cl.action_callback("Mecánica")
async def on_action_mecanica(action: cl.Action):
    module = action.payload.get("value", "General Physics")
    cl.user_session.set("module", module)
    await cl.Message(content=f"Módulo seleccionado: {module}").send()

@cl.action_callback("Electromagnetismo")
async def on_action_electromagnetismo(action: cl.Action):
    module = action.payload.get("value", "General Physics")
    cl.user_session.set("module", module)
    await cl.Message(content=f"Módulo seleccionado: {module}").send()

@cl.action_callback("Termodinámica")
async def on_action_termodinamica(action: cl.Action):
    module = action.payload.get("value", "General Physics")
    cl.user_session.set("module", module)
    await cl.Message(content=f"Módulo seleccionado: {module}").send()

@cl.action_callback("Física Cuántica")
async def on_action_fisica_cuantica(action: cl.Action):
    module = action.payload.get("value", "General Physics")
    cl.user_session.set("module", module)
    await cl.Message(content=f"Módulo seleccionado: {module}").send()

@cl.action_callback("General")
async def on_action_general(action: cl.Action):
    module = action.payload.get("value", "General Physics")
    cl.user_session.set("module", module)
    await cl.Message(content=f"Módulo seleccionado: {module}").send()

@cl.on_message
async def main(message: str):
    module = cl.user_session.get("module")
    prompt_module_text = f"Answer in the context of: {module}."
    session_id = cl.user_session.get("id")
    print(prompt_module_text)
    if module:
        response = process_question(message.content + prompt_module_text, session_id=session_id)
    else:
        response = process_question(message.content, session_id=session_id)
    await cl.Message(content=response).send()
