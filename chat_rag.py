import streamlit as st
import os
import pandas as pd
import time
import re
from langchain_core.tools import tool
from datetime import datetime

# --- TRUCO PARA CHROMA EN STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Asistente de Repuestos Pro", page_icon="⚙️")
st.title("⚙️ Sistema Inteligente de Repuestos")

# --- CONFIGURACIÓN DE API KEY (INTELIGENTE) ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("🔑 API Key de Gemini:", type="password")
    if not api_key:
        st.info("👈 Por favor, ingresa tu API Key para continuar.")
        st.stop()
else:
    st.sidebar.success("✅ API Key cargada desde Secrets")

# --- PANEL SECRETO PARA EL DUEÑO ---
with st.sidebar.expander("📦 Ver Pedidos Pendientes"):
    if os.path.exists("pedidos_pendientes.txt"):
        with open("pedidos_pendientes.txt", "r", encoding="utf-8") as f:
            contenido = f.read()
            st.text(contenido)
            
        # Un botón para limpiar la lista si ya se compraron
        if st.button("Borrar lista de pedidos"):
            os.remove("pedidos_pendientes.txt")
            st.rerun()
    else:
        st.info("Aún no hay piezas faltantes registradas.")

PERSIST_DIRECTORY = "db_catalogo_solo"

# --- HERRAMIENTAS DEL AGENTE (Nivel 3) ---
@tool
def registrar_pieza_faltante(pieza: str, vehiculo: str) -> str:
    """
    Útil EXCLUSIVAMENTE cuando el cliente busca un repuesto que NO está en el inventario.
    Guarda la solicitud en la lista de pedidos pendientes para el dueño del negocio.
    """
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("pedidos_pendientes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{fecha}] Solicitado: {pieza} - Para: {vehiculo}\n")
    return f"REGISTRO EXITOSO: La pieza '{pieza}' para '{vehiculo}' ha sido anotada para el proveedor."

# Lista de herramientas disponibles
lista_herramientas = [registrar_pieza_faltante]

# --- FUNCIONES DE DATOS ---
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
    ignore = ["para", "de", "el", "la", "un", "una", "del", "año", "años"]
    palabras = [p.lower() for p in re.findall(r'\w+', query) if p.lower() not in ignore]
    if not palabras: return ""
    resultado = df.copy()
    for p in palabras:
        mask = resultado.astype(str).apply(lambda x: x.str.contains(p, case=False, na=False)).any(axis=1)
        resultado = resultado[mask]
    if resultado.empty:
        importantes = [p for p in palabras if len(p) > 3]
        if importantes:
            mask = df.astype(str).apply(lambda x: x.str.contains('|'.join(importantes), case=False, na=False)).any(axis=1)
            resultado = df[mask]
    return resultado.head(15).to_string(index=False)

@st.cache_resource(show_spinner=False)
def preparar_catalogo(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    if os.path.exists("catalogo.pdf"):
        loader = PyPDFLoader("catalogo.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        bar = st.progress(0, text="Cargando catálogo técnico...")
        for i, d in enumerate(split_docs):
            vectorstore.add_documents([d])
            bar.progress((i+1)/len(split_docs))
            time.sleep(1.2)
        bar.empty()
        return vectorstore
    return None

# --- LÓGICA PRINCIPAL ---
inventario = cargar_inventario()

if api_key:
    try:
        base_datos = preparar_catalogo(api_key)
        if "mensajes" not in st.session_state: st.session_state.mensajes = []
        for msg in st.session_state.mensajes:
            with st.chat_message(msg["rol"]): st.markdown(msg["contenido"])

        pregunta = st.chat_input("¿Qué repuesto buscas?")
        
        if pregunta:
            st.chat_message("user").markdown(pregunta)
            
            # 1. Búsquedas previas
            info_csv = buscar_en_csv(inventario, pregunta) if inventario is not None else ""
            info_pdf = ""
            if base_datos:
                docs = base_datos.similarity_search(pregunta, k=3)
                info_pdf = "\n".join([d.page_content for d in docs])

            # 2. Configurar el Agente
            llm_base = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0, max_retries=3)
            # ATENCIÓN: Aquí atamos las herramientas
            llm_con_herramientas = llm_base.bind_tools(lista_herramientas)

            # 3. Preparar historial
            historial_texto = ""
            for m in st.session_state.mensajes[-4:]:
                rol = "Cliente" if m["rol"] == "user" else "Asistente"
                historial_texto += f"{rol}: {m['contenido']}\n"

            prompt = f"""Eres un experto en repuestos. 
            STOCK ACTUAL: {info_csv}
            DATOS TÉCNICOS: {info_pdf}
            HISTORIAL: {historial_texto}

            REGLAS:
            1. Si NO hay stock en el inventario ni en el catálogo, utiliza la herramienta 'registrar_pieza_faltante'.
            2. Presenta resultados en tablas Markdown.
            Pregunta: {pregunta}"""

            with st.spinner("Procesando..."):
                # La IA decide si usar herramienta o solo hablar
                res = llm_con_herramientas.invoke(prompt)
                
                # --- LÓGICA DE EJECUCIÓN DE HERRAMIENTAS ---
                respuesta_final = ""
                
                # ¿La IA quiere llamar a una herramienta?
                if res.tool_calls:
                    for call in res.tool_calls:
                        if call["name"] == "registrar_pieza_faltante":
                            # Ejecutamos la función real de Python
                            resultado_tool = registrar_pieza_faltante.invoke(call["args"])
                            respuesta_final = f"Lo siento, no tenemos ese repuesto. Pero no te preocupes: {resultado_tool}"
                else:
                    # Si no usa herramientas, sacamos el texto normal
                    if isinstance(res.content, str):
                        respuesta_final = res.content
                    elif isinstance(res.content, list):
                        for b in res.content:
                            if isinstance(b, dict) and "text" in b: respuesta_final += b["text"]

                st.chat_message("assistant").markdown(respuesta_final)
                st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
                st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta_final})

    except Exception as e:
        st.error(f"Error: {e}")
