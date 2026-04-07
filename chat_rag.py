import streamlit as st
import os
import pandas as pd
import time
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Asistente de Repuestos Pro", page_icon="⚙️")
st.title("⚙️ Sistema Inteligente de Repuestos")

api_key = st.sidebar.text_input("🔑 API Key de Gemini:", type="password")
PERSIST_DIRECTORY = "db_catalogo_solo"

# --- 1. FUNCIÓN DE BÚSQUEDA EN INVENTARIO (MEJORADA) ---
@st.cache_data
def cargar_inventario():
    if os.path.exists("inventario.csv"):
        try:
            df = pd.read_csv("inventario.csv", sep=";", encoding="latin-1")
            df.columns = df.columns.str.strip()
            return df
        except:
            return pd.read_csv("inventario.csv", sep=";", encoding="ISO-8859-1")
    return None

def buscar_en_csv(df, query):
    # Limpiamos la consulta de palabras vacías
    ignore = ["para", "de", "el", "la", "un", "una", "del", "año", "años"]
    palabras = [p.lower() for p in re.findall(r'\w+', query) if p.lower() not in ignore]
    
    if not palabras: return ""

    # LÓGICA "AND": Buscamos filas que contengan TODAS las palabras clave
    resultado = df.copy()
    for p in palabras:
        mask = resultado.astype(str).apply(lambda x: x.str.contains(p, case=False, na=False)).any(axis=1)
        resultado = resultado[mask]
    
    # Si el AND es muy estricto y no sale nada, intentamos con las palabras más largas (marcas/modelos)
    if resultado.empty:
        importantes = [p for p in palabras if len(p) > 3]
        if importantes:
            mask = df.astype(str).apply(lambda x: x.str.contains('|'.join(importantes), case=False, na=False)).any(axis=1)
            resultado = df[mask]

    return resultado.head(15).to_string(index=False)

# --- 2. PREPARAR CATÁLOGO PDF ---
@st.cache_resource(show_spinner=False)
def preparar_catalogo(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    if os.path.exists("catalogo.pdf"):
        loader = PyPDFLoader("catalogo.pdf")
        docs = loader.load()
        # Chunks específicos para tablas de catálogos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        # Carga lenta para evitar Error 429
        bar = st.progress(0, text="Cargando catálogo técnico...")
        for i, d in enumerate(split_docs):
            vectorstore.add_documents([d])
            bar.progress((i+1)/len(split_docs))
            time.sleep(1.2)
        bar.empty()
        return vectorstore
    return None

# --- INTERFAZ Y LÓGICA ---
inventario = cargar_inventario()

if api_key:
    try:
        base_datos = preparar_catalogo(api_key)
        
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []
            
        for msg in st.session_state.mensajes:
            with st.chat_message(msg["rol"]):
                st.markdown(msg["contenido"])

        pregunta = st.chat_input("¿Qué repuesto buscas?")
        
        if pregunta:
            st.chat_message("user").markdown(pregunta)
            
            # 1. Buscar en CSV (Stock)
            info_csv = buscar_en_csv(inventario, pregunta) if inventario is not None else ""
            
            # 2. Buscar en PDF (Técnico)
            info_pdf = ""
            if base_datos:
                docs = base_datos.similarity_search(pregunta, k=3)
                info_pdf = "\n".join([d.page_content for d in docs])

            # 3. Respuesta de la IA
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            prompt = f"""Eres un experto en repuestos de una tienda física. Tienes dos fuentes de datos:
            
            FUENTE 1: INVENTARIO FÍSICO (CSV) -> Aquí están los precios y existencias reales.
            {info_csv}
            
            FUENTE 2: CATÁLOGO TÉCNICO (PDF) -> Aquí están las especificaciones de bujías y compatibilidades.
            {info_pdf}
            
            REGLAS CRÍTICAS:
            1. Si el cliente pregunta por un año (ej: 98), busca en los rangos (ej: 93-98 incluye al 98).
            2. Si hay datos en el INVENTARIO, dáselos (Marca, Código, Existencia).
            3. Si el cliente pide bujías y no están en el inventario, revisa el CATÁLOGO TÉCNICO para dar el código NGK.
            4. Si NO encuentras el repuesto en ninguna de las dos fuentes, di que no hay stock.
            
            Pregunta del cliente: {pregunta}"""
            
            with st.spinner("Buscando..."):
                res = llm.invoke(prompt)
                st.chat_message("assistant").markdown(res.content)
                st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
                st.session_state.mensajes.append({"rol": "assistant", "contenido": res.content})
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Ingresa tu API Key.")