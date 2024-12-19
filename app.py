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
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="LangChain APP",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

#file = st.file_uploader("Upload a PDF file", accept_multiple_files=False, type="pdf")

source_data_folder = "./content/MisDatos"
# Leyendo los PDFs del directorio configurado
loader = PyPDFDirectoryLoader(source_data_folder)
data_on_pdf = loader.load()

# Particionando los datos. Con un tamaño delimitado (chunks) y 
# 200 caracters de overlapping para preservar el contexto
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], #para que corte las frases por ahi
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(data_on_pdf)

from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings

# Crea la instancia de embeddings con Cohere

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY_GOOGLE"])

path_db = "./content/VectorDB"  # Ruta a la base de datos del vector store

# Crear el vector store a partir de tus documentos 'splits'
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings_model, 
    persist_directory=path_db #esto lo que haces es guardarlo en el disco duro
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["API_KEY_GOOGLE"]) #para que esto funcione tiene que estra en los secretos

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt") #parte del langchain que recoge informacion para pasarsela al modelo

def format_docs(docs):
    # Funcion auxiliar para enviar el contexto al modelo como parte del prompt
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

pregunta = "Que es la pacobilla? " # @param {type:"string"}
response = rag_chain.invoke(pregunta )
print(response)

st.title("LangChain APP 📖🔎")
st.write("¡Bienvenidos a nuestra aplicación interactiva desarrollada con Streamlit! Esta herramienta te permite cargar un archivo PDF y, mediante procesamiento de lenguaje natural, puedes hacerle preguntas sobre el contenido del documento. Ya sea para extraer información clave o resolver dudas específicas, nuestra app facilita la interacción y comprensión de los archivos PDF de manera rápida y sencilla. ¡Explora el contenido de tus documentos de una forma más eficiente y accesible! ✌🏼")

pregunta = st.text_input("ASK:", "Ques es la pacobilla?")

if st.button ("RUN QUERY"):
    responde = rag_chain.invoke(pregunta)
    print(response)
    st.write(response) 