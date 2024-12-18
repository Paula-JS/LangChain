import streamlit as st
import os
from IPython.display import Markdown

# Librerías para la preparación de datos
from langchain.document_loaders import PyPDFDirectoryLoader #leer un pdf 
from langchain.text_splitter import RecursiveCharacterTextSplitter #para hacer chunks
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings #obtener embedings

# Librerías para el proceso de Retrieval
from langchain import hub #para traernos el prompt
from langchain_core.output_parsers import StrOutputParser #la respuesta que devuelve Gemini, no tengas que estar buscando el texto
from langchain_core.runnables import RunnablePassthrough 
from langchain_google_genai import ChatGoogleGenerativeAI #clase del LLM que estamos usando
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(
    page_title="LangChain APP",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LangChain APP 📖🔎")
st.write("¡Bienvenidos a nuestra aplicación interactiva desarrollada con Streamlit! Esta herramienta te permite cargar un archivo PDF y, mediante procesamiento de lenguaje natural, puedes hacerle preguntas sobre el contenido del documento. Ya sea para extraer información clave o resolver dudas específicas, nuestra app facilita la interacción y comprensión de los archivos PDF de manera rápida y sencilla. ¡Explora el contenido de tus documentos de una forma más eficiente y accesible! ✌🏼")

file = st.file_uploader("Inserta un documento PDF", type="pdf")

import PyPDF2

import PyPDF2

splits = []
if file is not None:
    # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(file)
    # Extract the content
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    # Display the content

    documents = [Document(page_content=content)]
    # 200 caracters de overlapping para preservar el contexto
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], #para que corte las frases por ahi
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
        
from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings

# Crea la instancia de embeddings con Cohere
embeddings_model = CohereEmbeddings(cohere_api_key=os.environ["API_KEY_COHERE"], user_agent="antonio")

path_db = "./content/VectorDB"  # Ruta a la base de datos del vector store

# Crear el vector store a partir de tus documentos 'splits'
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings_model, 
    persist_directory=path_db
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["API_KEY_GOOGLE"])

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    # Funcion auxiliar para enviar el contexto al modelo como parte del prompt
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

prompt.messages[0].prompt.template = "You are a Bolivian kitcken expert assistant for question-answering tasks (thus you might wanna answer in Spanish). Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"

user_input = st.text_input("Hazme una pregunta")


if st.button("RUN QUERY", type="primary"):
    if user_input:
        response = rag_chain.invoke(user_input)
        Markdown(response)

        st.write("Query: " + user_input)
        st.header("Generated Answer: " )
        st.write("Retrieved chunk: " + response)


