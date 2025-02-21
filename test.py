from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    context: str


from preprocessing import query_vector_database
import torch
import os
import dotenv
import getpass

dotenv.load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "pdf_fragments"
CHUNK_SIZE = 1000  # Número máximo de caracteres por fragmento


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages([ ("system", """
                    Eres un profesor de astrofísica. Responde la siguiente pregunta: {question}
                    Utiliza ÚNICAMENTE el contexto proporcionado a continuación (los documentos están en inglés) para elaborar tu respuesta. Responde en el idioma en el que se formule la pregunta, pero conserva en inglés las citas textuales y los metadatos que se encuentren en el contexto.
                    Si el contexto resulta insuficiente, ambiguo o no se proporciona, contesta:
                        "No dispongo de suficiente información para responder a la pregunta."
                    Además, en tu respuesta debes:
                        - Citar de forma estandarizada la(s) página(s) y fuente(s) según los metadatos incluidos en el contexto para cada fragmento utilizado.
                        - Indicar si existen contradicciones en el contexto y explicar brevemente cómo las has resuelto. Si no existen contradicciones, simplemente omítelo.
                    Contexto:
                    {context}
                    """), MessagesPlaceholder(variable_name="messages"),])


# Define a new graph
workflow = StateGraph(state_schema=State)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


while True:
    config = {"configurable": {"thread_id": "abc123"}}
    query = input("Text here:")
    context = query_vector_database(query)
    input_messages = [HumanMessage(query)]
    output = app.invoke(
    {"messages": input_messages, "context": context, "question": query},
    config,
)
    output["messages"][-1].pretty_print()  # output contains all messages in state

