import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from preprocessing import query_vector_database
import torch
import os
import dotenv

dotenv.load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "pdf_fragments"
CHUNK_SIZE = 1000  # Número máximo de caracteres por fragmento

def get_answer(question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("openai_key"))
    prompt_template = """
    Eres un profesor de astrofísica. Responde la siguiente pregunta: {question}
    Utiliza ÚNICAMENTE el contexto proporcionado a continuación (los documentos están en inglés) para elaborar tu respuesta. Responde en el idioma en el que se formule la pregunta, pero conserva en inglés las citas textuales y los metadatos que se encuentren en el contexto.
    Si el contexto resulta insuficiente, ambiguo o no se proporciona, contesta:
        "No dispongo de suficiente información para responder a la pregunta."
    Además, en tu respuesta debes:
        - Citar de forma estandarizada la(s) página(s) y fuente(s) según los metadatos incluidos en el contexto para cada fragmento utilizado.
        - Indicar si existen contradicciones en el contexto y explicar brevemente cómo las has resuelto. Si no existen contradicciones, simplemente omítelo.
    Contexto:
    {context}
    """
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    chain = prompt | llm | StrOutputParser()
    # Obtiene el contexto a partir de la pregunta (ya que query_vector_database espera una cadena)
    context = query_vector_database(question)
    response = chain.invoke({"question": question, "context": context})
    return response

@cl.on_chat_start
async def start():
    # En versiones recientes se utiliza cl.Message y send() de forma asíncrona
    await cl.Message(content="¡Hola! Soy tu profesor de astrofísica. ¿En qué puedo ayudarte hoy?").send()

@cl.on_message
async def main(message: cl.Message):
    # Extraemos el texto de la pregunta usando message.content
    answer = get_answer(message.content)
    await cl.Message(content=answer).send()
