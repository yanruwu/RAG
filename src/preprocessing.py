import os
import asyncio
import torch
from tqdm import tqdm
import spacy
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.docstore.document import Document  # Para crear nuevos documentos

# Variables de configuración global
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "pdf_fragments"
CHUNK_SIZE = 1000  # Número máximo de caracteres por fragmento

# Cargar el modelo de spaCy para el chunking (si no está instalado, se descarga)
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' not found. Downloading now...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def spacy_chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Divide el texto en fragmentos basándose en la segmentación de oraciones de spaCy.
    
    Args:
        text (str): Texto a dividir.
        chunk_size (int): Número máximo de caracteres por fragmento.
        
    Returns:
        List[str]: Lista de fragmentos.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Si al agregar la oración no se supera el límite, se concatena
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            # Se guarda el fragmento actual y se inicia uno nuevo con la oración
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def is_low_semantic_content(text: str, token_threshold: float = 0.35, min_sentences: int = 2, min_avg_tokens: int = 5) -> bool:
    """
    Determina si un fragmento de texto tiene bajo contenido semántico.
    
    Args:
        text (str): Fragmento de texto a evaluar.
        token_threshold (float): Proporción mínima de tokens significativos.
        min_sentences (int): Número mínimo de oraciones.
        min_avg_tokens (int): Número mínimo promedio de tokens por oración.
        
    Returns:
        bool: True si el fragmento es considerado de bajo contenido, False en caso contrario.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # Si hay pocas oraciones, puede ser un índice o anexo
    if len(sentences) < min_sentences:
        return True

    # Filtrar tokens significativos (excluyendo puntuación y números)
    total_tokens = [token for token in doc if not token.is_space]
    significant_tokens = [token for token in total_tokens if not token.is_punct and not token.like_num]

    if not total_tokens:
        return True  # Fragmento vacío

    ratio_significant = len(significant_tokens) / len(total_tokens)
    avg_tokens_per_sentence = sum(len([t for t in sent if not t.is_space]) for sent in sentences) / len(sentences)

    if ratio_significant < token_threshold or avg_tokens_per_sentence < min_avg_tokens:
        return True

    return False


def nlp_split_documents(documentos, chunk_size: int = CHUNK_SIZE):
    """
    Aplica el chunking basado en spaCy a cada documento, filtra fragmentos con bajo contenido semántico
    y devuelve una lista de nuevos Document.
    
    Args:
        documentos (List[Document]): Documentos originales.
        chunk_size (int): Número máximo de caracteres por fragmento.
        
    Returns:
        List[Document]: Lista de documentos con fragmentos generados.
    """
    new_docs = []
    for doc in documentos:
        # Obtener el número de página del documento (suponiendo que esté en la metadata)
        page = doc.metadata.get("page", "N/A")
        # Generar fragmentos usando spaCy
        chunks = spacy_chunk_text(doc.page_content, chunk_size=chunk_size)
        for chunk in chunks:
            # Descartar fragmentos con bajo contenido semántico
            if is_low_semantic_content(chunk):
                continue
            # Crear un nuevo documento conservando la metadata (aquí solo la página)
            new_docs.append(Document(page_content=chunk, metadata={"page": page}))
    return new_docs


def preprocess_pdf_directory(pdf_directory: str) -> None:
    """
    Preprocesa todos los documentos PDF en un directorio: carga, divide en fragmentos (usando spaCy para chunking),
    limpia el texto, genera embeddings y almacena la información en ChromaDB.
    
    Args:
        pdf_directory (str): Ruta al directorio con archivos PDF.
    """
    # Inicializar el cliente persistente de ChromaDB y la colección
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Listar todos los archivos PDF en el directorio
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print("No se encontraron archivos PDF en el directorio.")
        return
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)

        # Normalizamos la ruta para que sea igual que en las metadatas
        normalized_path = pdf_path.replace("\\", "/")        
        existing_embeddings = collection.get(
            where={"source": normalized_path},
            include=["documents"]
        )
        if len(existing_embeddings["documents"]) > 0:
            print(f"Embeddings ya existen para {pdf_file}. Se omite el procesamiento.\n")
            continue
        else:
            print(f"Procesando el archivo: {pdf_file} ...")
        
        # Función asíncrona para cargar el PDF
        async def cargar_pdf(ruta: str):
            print(f"Cargando el PDF: {ruta} ...")
            loader = PyPDFLoader(ruta)
            documentos = await loader.aload()
            print(f"PDF cargado: {ruta}.")
            return documentos

        documentos = asyncio.run(cargar_pdf(pdf_path))

        # Aplicar chunking basado en spaCy a cada documento y filtrar fragmentos irrelevantes
        print("Dividiendo el contenido en fragmentos utilizando spaCy...")
        fragmentos = nlp_split_documents(documentos, chunk_size=CHUNK_SIZE)
        print(f"Se han generado {len(fragmentos)} fragmentos útiles.")

        # Limpiar fragmentos: reemplazar saltos de línea por espacios
        print("Limpiando fragmentos (reemplazando saltos de línea)...")
        for frag in tqdm(fragmentos, desc="Limpiando fragmentos", unit="fragmento"):
            frag.page_content = frag.page_content.replace("\n", " ")

        # Extraer textos y preparar metadatos (incluyendo el número de página)
        texts = [frag.page_content for frag in fragmentos]
        metadatas = []
        for frag in fragmentos:
            meta = {"source": normalized_path, "page": frag.metadata.get("page", "N/A")}
            metadatas.append(meta)

        print(f"\nUsando dispositivo: {DEVICE}")

        # Inicializar el modelo de embeddings
        embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)

        print("\nGenerando embeddings...")
        embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

        # Generar IDs únicos para cada fragmento
        ids = [f"{pdf_file}_doc_{i}" for i in range(len(texts))]

        print("Agregando documentos a la colección de ChromaDB...")
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Proceso completado para el archivo: {pdf_file}. Los embeddings se han generado y almacenado.\n")

    print("Proceso completado para todos los archivos en el directorio.")


def query_vector_database(query: str) -> list:
    """
    Realiza una consulta sobre la base de datos vectorial y muestra los fragmentos
    más similares a la query ingresada.
    
    Args:
        query (str): El prompt o consulta a buscar.
    
    Returns:
        list: Lista de fragmentos relevantes con metadatos.
    """
    # Inicializar el cliente persistente y la colección
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Inicializar el modelo de embeddings
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    query_embedding = embedding_model.encode(query, show_progress_bar=False).tolist()

    resultados = collection.query(
        query_embeddings=[query_embedding],
        n_results=15,
        include=["documents", "metadatas", "distances"]
    )

    results = []
    for idx, (doc, meta) in enumerate(zip(
            resultados["documents"][0],
            resultados["metadatas"][0])):
        results.append(doc + " || FRAGMENT INFO: " + str(meta) + "||")
    return results


def obtener_todos_los_sources() -> set:
    """
    Obtiene todos los sources (rutas de archivos PDF) almacenados en la colección de ChromaDB.
    
    Returns:
        set: Un conjunto de sources únicos.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    datos = collection.get(include=["metadatas"])
    
    sources = set()
    for metadata in datos["metadatas"]:
        for meta in metadata:
            source = meta.get("source")
            if source:
                sources.add(source)
    
    print("\nSources almacenados en la colección:")
    for source in sorted(sources):
        print(source)
    
    return sources
