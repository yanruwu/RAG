# Proyecto RAG – ANStein 🚀

Este proyecto es una implementación de un sistema **Retrieval-Augmented Generation (RAG)** enfocado en el ámbito de la docencia en física. Utiliza técnicas de recuperación y generación de respuestas para ayudar a resolver dudas o explicar conceptos en áreas específicas de la física, tales como **Mecánica**, **Electromagnetismo**, **Termodinámica**, **Física Cuántica** y **Física General**. 🤓

El sistema combina la descarga y el preprocesamiento de documentos (PDFs) con la generación de respuestas usando modelos de lenguaje (utilizando *langchain*, *ChatOpenAI* y *chainlit*), junto con el almacenamiento y consulta de embeddings en una base de datos vectorial (*ChromaDB*).

---

## Características ✨

- **Descarga y Preprocesamiento de Documentos:**  
  Descarga automáticamente PDFs listados en `urls.txt`, los preprocesa y los fragmenta en secciones útiles para la búsqueda y recuperación de información. 📄➡️📚

- **Consulta y Respuesta Dinámica:**  
  Al iniciar la aplicación, se invita al usuario a seleccionar un módulo (por ejemplo, *Mecánica*, *Electromagnetismo*, etc.) y posteriormente se responde a las preguntas en el contexto del área seleccionada. 💬✅

- **Historial de Conversaciones:**  
  Se mantiene un historial en memoria para mejorar la interacción y el contexto en conversaciones posteriores. 🗂️

- **Integración con Chainlit:**  
  La interfaz de usuario se basa en Chainlit, que permite una interacción sencilla y visual con la aplicación. 🎨

- **Requiere Clave de OPENAI:**  
  Es **necesario** disponer de una clave de OPENAI para el correcto funcionamiento del modelo de lenguaje. Esta clave debe ser alojada en un archivo `.env` al crear el proyecto. 🔑

---

## Estructura del Proyecto 📁
```
└── yanruwu-rag/
    ├── README.md
    ├── chainlit.md          # Guía de usuario para la interfaz Chainlit.
    ├── main.py              # Script principal: descarga, preprocesamiento y chat.
    ├── requirements.yml     # Lista de dependencias.
    ├── urls.txt             # URLs de los PDFs a descargar.
    ├── chroma_db/          # Base de datos vectorial (ChromaDB).
    ├── public/             # Recursos públicos.
    ├── src/                # Código fuente del proyecto.
    │   ├── doc_load.py      # Descarga de PDFs.
    │   ├── memory_chat.py   # Configuración del chat y manejo del historial.
    │   └── preprocessing.py # Preprocesamiento: fragmentación, limpieza y generación de embeddings.
    └── .chainlit/          # Configuración de Chainlit.
        ├── config.toml      # Ajustes de la interfaz.
        └── translations/    # Archivos de traducción de chainlit.
```

---

## Instalación 💻

1. **Clonar el repositorio:**

```bash
git clone https://github.com/yanruwu/RAG.git
cd RAG
```
2. **Crear el entorno de Conda y activar:**
```bash
conda env create -f requirements.yml
conda activate anstein-rag
```
3. **Instalar las dependencias de pip:**

Las dependencias listadas en requirements.yml se instalarán automáticamente al crear el entorno. Si usas otro gestor, asegúrate de instalar:
- pypdf
- chainlit
- langchain
- deep_translator
- langdetect
- sentence-transformers
- chromadb
- langchain_openai
- langchain_community
4. **Configurar la clave de OPENAI:**
Crea un archivo .env en la raíz del proyecto y añade tu clave de OPENAI:
```env
OPENAI_API_KEY=tu_clave_de_openai_aqui
```
---
## Uso 🔧

1. **Descarga y Preprocesamiento:**

Al ejecutar main.py, se realizará lo siguiente:

- Se leen las URLs de urls.txt y se descargan los documentos PDF (almacenados en el directorio docs).
- Se preprocesan los PDFs: se dividen en fragmentos basados en oraciones usando SpaCy, se generan embeddings con SentenceTransformer y se almacenan en ChromaDB.

2. **Inicio del Chat:**

Al iniciar la aplicación, Chainlit mostrará un mensaje de bienvenida con botones para seleccionar un módulo (por ejemplo, Mecánica, Electromagnetismo, etc.). Una vez seleccionado, el usuario podrá:

- Enviar preguntas relacionadas con el área de física seleccionada.
- El sistema traducirá (si es necesario), recuperará el contexto relevante y generará una respuesta utilizando el modelo configurado. 🤖
---
## Configuración ⚙️
- **Chainlit:**
El archivo .chainlit/config.toml contiene la configuración de la interfaz.

- **Embeddings y Base de Datos Vectorial:**
La configuración de preprocesamiento y consulta se define en ``src/preprocessing.py``, donde se especifica el directorio de persistencia para ChromaDB y el modelo de embeddings a utilizar.

- **Modelos y API:**
En ``src/memory_chat.py`` se configura el modelo de lenguaje (por ejemplo, ChatOpenAI con modelo ``gpt-4o-mini-2024-07-18``).
---

## Ejemplo
El inicio de chat da la opción de elegir el área concreta para basar la respuesta.

![Inicio de chat](/public/imgs/test1.png)

Podemos seguir conversando con el chatbot, el cual mantiene la memoria de la conversación:

![Demostración de memoria](/public/imgs/test2.png)

Además, las expresiones matemáticas se formatean con LaTeX, lo que permite una muestra clara y entendible. El modelo también provee las fuentes de donde saca la información para permitir al usuario consultarlas por sí mismo.

---

## Contribuciones 🤝
Si deseas contribuir al proyecto, siéntete libre de abrir issues o pull requests. Las contribuciones para mejorar la funcionalidad, la documentación o la traducción a otros idiomas serán muy bien recibidas.



