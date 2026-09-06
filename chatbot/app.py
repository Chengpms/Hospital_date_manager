"""
Módulo de la Interfaz de Usuario utilizando Streamlit.
Actúa exclusivamente como capa de presentación, delegando toda la lógica
al ConversationManager y al Predictor.
"""

import logging
import streamlit as st
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from chatbot.config import LLM_CONFIG, APP_CONFIG, PREDICT_CONFIG
from chatbot.providers.provider_factory import ProviderFactory
from chatbot.conversation.conversation_manager import ConversationManager
from chatbot.prediction.predictor import Predictor

# --- BYPASS HACKATHON: Comentamos esto porque falta el archivo en Git ---
# from exports import (
#     export_conversation_json,
#     export_conversation_csv,
#     export_full_data_csv,
#     save_clinical_history,
# )
# ------------------------------------------------------------------------

# Configuración inicial de la página de Streamlit
st.set_page_config(
    page_title="Asistente de Admisión Hospitalaria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

def initialize_session() -> None:
    """
    Inicializa los objetos principales en la sesión de Streamlit si no existen.
    Garantiza la persistencia del estado entre recargas de la UI.
    """
    if "manager" not in st.session_state:
        logger.info("Inicializando nueva sesión de ConversationManager.")
        try:
            provider = ProviderFactory.get_provider()
            st.session_state.manager = ConversationManager(provider)
        except Exception as e:
            st.error(f"Error de configuración del proveedor LLM: {str(e)}")
            st.stop()
    else:
        if not hasattr(st.session_state.manager, "process_clinical_history"):
            logger.warning("Recreando ConversationManager porque falta process_clinical_history.")
            try:
                provider = ProviderFactory.get_provider()
                st.session_state.manager = ConversationManager(provider)
            except Exception as e:
                st.error(f"Error de configuración del proveedor LLM: {str(e)}")
                st.stop()
            
    if "predictor" not in st.session_state:
        st.session_state.predictor = Predictor()
        
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = [
            {"role": "assistant", "content": "Hola. Soy el asistente virtual del hospital. ¿En qué te puedo ayudar hoy?"}
        ]
    if "clinical_history_path" not in st.session_state:
        st.session_state.clinical_history_path = None

def reset_conversation() -> None:
    """Borra el estado actual para iniciar un nuevo flujo conversacional."""
    logger.info("Reiniciando la conversación a petición del usuario.")
    for key in ["manager", "predictor", "messages_ui", "clinical_history_path"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def render_sidebar() -> None:
    """
    Renderiza el panel lateral con las configuraciones técnicas.
    """
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")

        provider_choice = st.selectbox(
            "Proveedor LLM",
            options=["Ollama", "Gemini", "OpenAI"],
            index=0 if LLM_CONFIG.default_provider == "ollama" else (1 if LLM_CONFIG.default_provider == "gemini" else 2)
        )
        new_provider = provider_choice.lower()
        if new_provider != LLM_CONFIG.default_provider:
            LLM_CONFIG.default_provider = new_provider
            if "manager" in st.session_state:
                del st.session_state["manager"]
            st.rerun()

        LLM_CONFIG.temperature = st.slider(
            "Temperatura (Creatividad vs Precisión)",
            min_value=0.0, max_value=1.0, value=LLM_CONFIG.temperature, step=0.1,
            help="Mantenlo en 0.0 para maximizar la consistencia del JSON."
        )
        
        LLM_CONFIG.max_tokens = st.number_input(
            "Max Tokens", min_value=64, max_value=4096, value=LLM_CONFIG.max_tokens, step=64
        )
        
        st.divider()

        # Manual API credentials UI
        st.subheader("🔑 Configuración manual de API")
        use_manual = st.checkbox("Usar credenciales manuales (pegar clave/endpoint)")
        if use_manual:
            manual_provider = st.selectbox("Proveedor manual", ["Ollama","Gemini","OpenAI","Custom"], index=0)
            manual_api_key = st.text_input("API Key (si aplica)", value="", type="password")
            manual_base_url = st.text_input("Base URL / Endpoint (si aplica)", value="")
            manual_model = st.text_input("Nombre de Modelo (ej. gpt-4o-mini)", value="")
            manual_timeout = st.number_input("Timeout (segundos)", min_value=5, max_value=600, value=LLM_CONFIG.request_timeout)
            if st.button("Aplicar credenciales manuales"):
                LLM_CONFIG.default_provider = manual_provider.lower()
                if manual_api_key:
                    # try to set most common keys
                    setattr(LLM_CONFIG, 'gemini_api_key', manual_api_key)
                    setattr(LLM_CONFIG, 'openai_api_key', manual_api_key)
                if manual_base_url:
                    setattr(LLM_CONFIG, 'ollama_base_url', manual_base_url)
                    setattr(LLM_CONFIG, 'openai_base_url', manual_base_url)
                LLM_CONFIG.request_timeout = int(manual_timeout)

                # After applying creds, attempt to discover models automatically
                try:
                    provider = ProviderFactory.get_provider()
                    models = provider.list_models()
                    if models:
                        selected = st.selectbox("Modelos detectados", models, index=0)
                        LLM_CONFIG.default_model = selected
                        # provider-specific
                        if LLM_CONFIG.default_provider == 'gemini':
                            LLM_CONFIG.gemini_model = selected
                        if LLM_CONFIG.default_provider == 'openai':
                            LLM_CONFIG.openai_model = selected
                        st.success(f"Modelos detectados y seleccionado: {selected}")
                    else:
                        if manual_model:
                            LLM_CONFIG.default_model = manual_model
                            LLM_CONFIG.gemini_model = manual_model
                            LLM_CONFIG.openai_model = manual_model
                            st.warning("No se pudieron listar modelos automáticamente; se aplica el modelo que pegaste.")
                        else:
                            st.warning("No se encontraron modelos automáticamente. Introduce el nombre manualmente.")
                except Exception as e:
                    st.error(f"No se pudo inicializar el proveedor para listar modelos: {e}")
                    if manual_model:
                        LLM_CONFIG.default_model = manual_model
                        LLM_CONFIG.gemini_model = manual_model
                        LLM_CONFIG.openai_model = manual_model

                if "manager" in st.session_state:
                    del st.session_state["manager"]
                st.rerun()

        st.divider()
        if st.button("🔄 Nueva Conversación", use_container_width=True):
            reset_conversation()

def render_patient_status(state_dump: Dict[str, Any], missing_fields: list[str]) -> None:
    """
    Renderiza la tabla de estado del paciente en el panel lateral derecho.
    """
    st.subheader("📋 Estado del Paciente")
    
    if not state_dump:
        st.info("Aún no se ha recopilado información.")
        return

    extracted_data = {k: v for k, v in state_dump.items() if v.get("value") is not None}
    
    if extracted_data:
        for key, data in extracted_data.items():
            value = data["value"]
            conf = data["confidence"]
            color = "green" if conf >= 0.8 else ("orange" if conf >= 0.5 else "red")
            
            st.markdown(
                f"**{key}**: {value} <span style='color:{color}; font-size:0.8em;'>(Confianza: {conf:.2f})</span>", 
                unsafe_allow_html=True
            )
    
    st.divider()
    st.subheader("🎯 Variables Pendientes")
    if missing_fields:
        for field in missing_fields:
            st.markdown(f"- `{field}`")
    else:
        st.success("¡Información completada!")

def main() -> None:
    """Función principal que orquesta la construcción de la interfaz."""
    initialize_session()
    render_sidebar()
    
    header_col, status_col = st.columns([7, 3])
    with header_col:
        st.title("🏥 Asistente de Admisión Hospitalaria")
        st.caption("Interfaz de extracción conversacional y predicción de No-Show")

    model_path = Path(PREDICT_CONFIG.model_path)
    model_present = model_path.exists()
    with status_col:
        st.metric("LLM Provider", LLM_CONFIG.default_provider.capitalize())
        if model_present:
            st.success(f"Modelo: {model_path.name}")
        else:
            st.warning("Modelo ausente — se usará fallback matemático")

    col_chat, col_status = st.columns([7, 3])
    
    with col_chat:
        st.header("💬 Consulta Virtual")
        
        for msg in st.session_state.messages_ui:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            st.session_state.messages_ui.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analizando respuesta..."):
                    manager: ConversationManager = st.session_state.manager
                    reply, is_ready = manager.process_user_input(prompt)
                    
                    st.markdown(reply)
                    st.session_state.messages_ui.append({"role": "assistant", "content": reply})

    with col_status:
        manager = st.session_state.manager
        current_state = manager.get_current_state()
        missing = manager.state.get_missing_critical_fields()
        
        render_patient_status(current_state, missing)
        
        if manager.state.is_ready_for_prediction():
            st.divider()
            st.subheader("⚠️ Predicción de Riesgo (No-Show)")
            with st.spinner("Calculando probabilidad mediante XGBoost..."):
                predictor: Predictor = st.session_state.predictor
                result = predictor.predict(manager.state)
                
                risk_color = "red" if result.risk_level == "ALTO" else ("orange" if result.risk_level == "MEDIO" else "green")
                
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(0,0,0,0.1); border-left: 5px solid {risk_color};'>"
                    f"<h4>Nivel de Riesgo: {result.risk_level}</h4>"
                    f"<p>Probabilidad de ausencia: <strong>{result.probability:.2%}</strong></p>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
                if result.is_fallback:
                    st.caption("Nota: Se utilizó el modelo matemático de respaldo.")

        # --- BYPASS HACKATHON: Cortamos la ejecución de la interfaz aquí ---
        st.divider()
        st.info("⚡ Botones de exportación e historial desactivados temporalmente para evitar errores.")
        return 
        # -------------------------------------------------------------------

        # (El resto del código de exportación sigue abajo, pero Python lo ignorará gracias al return)
        st.divider()
        st.subheader("💾 Export Conversation")
        fmt = st.selectbox("Format", ("JSON", "CSV"))
        anonymize = st.checkbox("Anonymize (redact emails/phones/IDs)", value=True)

        st.markdown("**📎 Clinical History**")
        uploaded = st.file_uploader("Upload clinical history (PDF / TXT)", type=["pdf", "txt", "doc", "docx"], key="ch_upload")
        ch_text = st.text_area("Or paste clinical history text (optional)", height=120, key="ch_text")

        current_ch = st.session_state.get("clinical_history_path")
        if current_ch:
            st.info(f"Saved clinical history: {Path(current_ch).name}")

        col_btn, col_hist = st.columns([1, 2])
        with col_btn:
            if st.button("Create export", key="create_export"):
                pass

if __name__ == "__main__":
    main()