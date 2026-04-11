import streamlit as st
import os
import time
import sqlite3
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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

# --- CONFIGURACIÓN DE API KEY ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("🔑 API Key de Gemini:", type="password")
    if not api_key:
        st.info("👈 Por favor, ingresa tu API Key para continuar.")
        st.stop()
else:
    st.sidebar.success("✅ API Key cargada desde Secrets")

PERSIST_DIRECTORY = "db_catalogo_solo"

# ==========================================
# 🚀 HERRAMIENTAS DEL AGENTE (NUEVAS)
# ==========================================

@tool
def consultar_inventario_sql(consulta_sql: str) -> str:
    """
    Útil para buscar repuestos en el sistema de inventario.
    Recibe una consulta SQL válida (SELECT) para la tabla 'repuestos'.
    Si la consulta usa LIKE, recuerda usar los comodines %.
    Devuelve los registros encontrados.
    """
    try:
        conexion = sqlite3.connect("inventario.db")
        cursor = conexion.cursor()
        cursor.execute(consulta_sql)
        resultados = cursor.fetchall()
        columnas = [desc[0] for desc in cursor.description]
        conexion.close()
        
        if not resultados:
            return "No se encontraron resultados en la base de datos."
        
        # Formateamos para que la IA lo entienda fácil
        texto = f"Columnas: {columnas}\n"
        for fila in resultados[:15]: # Límite de 15 para no saturar
            texto += str(fila) + "\n"
        return texto
    except Exception as e:
        return f"Error en SQL: {e}"

@tool
def registrar_pieza_faltante(pieza: str, vehiculo: str) -> str:
    """
    Útil EXCLUSIVAMENTE cuando buscas un repuesto con la herramienta SQL y NO HAY RESULTADOS.
    Guarda la solicitud para el dueño del negocio.
    """
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("pedidos_pendientes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{fecha}] Solicitado: {pieza} - Para: {vehiculo}\n")
    return "Registro exitoso en pedidos pendientes."

# Agrupamos las herramientas
lista_herramientas = [consultar_inventario_sql, registrar_pieza_faltante]
diccionario_herramientas = {
    "consultar_inventario_sql": consultar_inventario_sql,
    "registrar_pieza_faltante": registrar_pieza_faltante
}


# ==========================================
# 📄 PREPARACIÓN DE CATÁLOGO PDF (RAG)
# ==========================================
@st.cache_resource(show_spinner=False)
def preparar_catalogo(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    # Aquí puedes actualizar al modelo de embedding nuevo en el futuro si quieres
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


# ==========================================
# 🧠 INTERFAZ Y LÓGICA PRINCIPAL
# ==========================================
if api_key:
    try:
        base_datos = preparar_catalogo(api_key)
        if "mensajes" not in st.session_state: st.session_state.mensajes = []
        for msg in st.session_state.mensajes:
            with st.chat_message(msg["rol"]): st.markdown(msg["contenido"])

        pregunta = st.chat_input("¿Qué repuesto buscas?")
        
        if pregunta:
            st.chat_message("user").markdown(pregunta)
            
            # 1. Buscar en PDF (El PDF sigue usando RAG clásico)
            info_pdf = ""
            if base_datos:
                docs = base_datos.similarity_search(pregunta, k=3)
                info_pdf = "\n".join([d.page_content for d in docs])

            # 2. Configurar el Agente
            llm_base = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0, max_retries=3)
            llm_con_herramientas = llm_base.bind_tools(lista_herramientas)

            # 3. Preparar historial
            historial_texto = ""
            for m in st.session_state.mensajes[-4:]:
                rol = "Cliente" if m["rol"] == "user" else "Asistente"
                historial_texto += f"{rol}: {m['contenido']}\n"

            # 4. El Prompt de Sistema
            instrucciones = f"""Eres un experto en repuestos. 
            DATOS TÉCNICOS (PDF): {info_pdf}
            HISTORIAL: {historial_texto}

            REGLAS:
            1. Para saber si hay repuestos, DEBES usar la herramienta 'consultar_inventario_sql'.
            2. La base de datos tiene una tabla llamada 'repuestos'. 
            3. Haz consultas seguras (ej: SELECT * FROM repuestos WHERE descripcion LIKE '%bujia%').
            4. Cuando tengas los datos de la base de datos, preséntalos en una tabla Markdown bonita.
            5. Si tras usar SQL no hay resultados, usa la herramienta 'registrar_pieza_faltante'.
            """

            with st.spinner("Procesando..."):
                # --- EL CICLO DE RAZONAMIENTO DEL AGENTE ---
                mensajes_conversacion = [
                    SystemMessage(content=instrucciones),
                    HumanMessage(content=pregunta)
                ]
                
                # Paso 1: La IA piensa y decide qué herramienta usar
                respuesta_ia = llm_con_herramientas.invoke(mensajes_conversacion)
                mensajes_conversacion.append(respuesta_ia)
                
                # Paso 2: Ejecutamos las herramientas que la IA pidió
                if respuesta_ia.tool_calls:
                    for call in respuesta_ia.tool_calls:
                        nombre_funcion = call["name"]
                        argumentos = call["args"]
                        
                        # Ejecutar la función Python real
                        funcion_real = diccionario_herramientas[nombre_funcion]
                        resultado_herramienta = funcion_real.invoke(argumentos)
                        
                        # Le pasamos el resultado de vuelta a la IA
                        mensajes_conversacion.append(
                            ToolMessage(content=str(resultado_herramienta), tool_call_id=call["id"])
                        )
                    
                    # Paso 3: La IA lee los datos de la base de datos y redacta la respuesta final
                    respuesta_final_ia = llm_con_herramientas.invoke(mensajes_conversacion)
                    texto_final = respuesta_final_ia.content
                else:
                    texto_final = respuesta_ia.content

                # Mostramos y guardamos
                st.chat_message("assistant").markdown(texto_final)
                st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
                st.session_state.mensajes.append({"rol": "assistant", "contenido": texto_final})

    except Exception as e:
        st.error(f"Error: {e}")
