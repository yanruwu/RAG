from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

from deep_translator import GoogleTranslator

from langdetect import detect
from src.preprocessing import query_vector_database

def init_chain_components():
    """
    Inicializa el modelo, la plantilla del prompt y el objeto chain con historial.
    Devuelve el objeto chain_with_history.
    """
    # Inicializa el modelo de lenguaje
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    
    # Crea la plantilla del prompt con el mensaje del sistema, historial y mensaje del usuario
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                """Eres un profesor universitario. Responde a la pregunta siguiendo estas instrucciones:
                1. Contexto Exclusivo:
                Utiliza ÚNICAMENTE el contexto proporcionado a continuación (los documentos están en inglés) para elaborar tu respuesta.
                2. Citas y Metadatos:
                Conserva en inglés las citas textuales y los metadatos que se encuentren en el contexto.
                Cita de forma estandarizada la(s) página(s) y fuente(s) según dichos metadatos para cada fragmento utilizado.
                3. Cálculos y Expresiones Matemáticas:
                Realiza cálculos si hay expresiones matemáticas en el contexto que puedan ayudarte a responder la pregunta, siempre citando la fuente de las expresiones matemáticas.
                Por favor, muestra las ecuaciones matemáticas usando el formato LaTeX en modo display, es decir, encerrándolas entre $$, mientras que en inline mode se encierran entre $.
                Ejemplo: $$x^2 + y^2 = z^2$$ para display, $x^2 + y^2 = z^2$ para inline.
                4. Respuestas Basadas en el Contexto:
                - Si el contexto es insuficiente, ambiguo o no se proporciona, responde: "No dispongo de suficiente información para responder a la pregunta."
                - Si la pregunta no guarda relación con el contexto, responde: "La pregunta no está relacionada con el contexto proporcionado."
                5. Contradicciones:
                Si existen contradicciones en el contexto, indícalo y explica brevemente cómo las has resuelto. Si no existen, omite este paso.
                
                Contexto:
                {context}"""
            )
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    # Crea la cadena (chain) conectando el prompt, el modelo y el parser de salida
    chain = prompt_template | model | StrOutputParser()
    
    # Diccionario para almacenar el historial por sesión
    store = {}
    
    def get_by_session_id(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # Crea el objeto chain_with_history que gestiona el historial de mensajes
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    return chain_with_history

# Inicializamos el chain con historial una sola vez
chain_with_history = init_chain_components()

def process_question(user_question: str, session_id: str = "foo") -> str:
    """
    Procesa la pregunta del usuario:
      - Detecta el idioma de la pregunta.
      - Agrega un prompt que indique en qué idioma se debe responder.
      - Consulta el contexto a través de query_vector_database.
      - Invoca el chain con historial y retorna la respuesta.
    """
    # Detecta el idioma de la pregunta
    language = detect(user_question)
    print(language)
    if language == "es": 
        # Traduce la pregunta al inglés para mejor búsqueda con embedding (así nos ahorramos tener que usar un modelo multilingüe)
        user_question = GoogleTranslator(source=language, target="en").translate(user_question)
    language_prompt = f"Answer in {language}:"
    final_question = f"{language_prompt} {user_question}. Remember the math formatting rules."
    
    # Consulta el contexto (por ejemplo, documentos relacionados)
    context = query_vector_database(final_question)
    
    # Invoca el chain con historial
    result = chain_with_history.invoke(
        {"question": final_question, "context": context},
        config={"configurable": {"session_id": session_id}},
    )
    return result

if __name__ == "__main__":
    while True:
        user_question = input(">>> ")
        response = process_question(user_question)
        print(response)
