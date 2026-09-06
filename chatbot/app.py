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

# Custom CSS for visual polish
st.markdown(
    """
    <style>
    .chat-bubble-user { background:#e6f2ff; padding:10px; border-radius:10px; margin:6px 0;}
    .chat-bubble-assistant { background:#f1f8e9; padding:10px; border-radius:10px; margin:6px 0;}
    .risk-badge { padding:8px 12px; border-radius:6px; color:#fff; font-weight:600; display:inline-block;}
    .risk-high { background: #d32f2f; }
    .risk-medium { background: #f57c00; }
    .risk-low { background: #2e7d32; }
    .small-note { font-size:0.9em; color:#666; }
    .sidebar-section { margin-bottom: 12px; }
    </style>
    """,
    unsafe_allow_html=True
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
    """Función principal: interfaz rediseñada, llamada a proveedores y experiencia mejorada."""
    st.title("🏥 Asistente de Admisión Hospitalaria — Chatbot")
    st.markdown("Una interfaz limpia para ingresar credenciales de LLM, descubrir modelos y mantener conversaciones estructuradas.")

    # Sidebar: provider + creds + model discovery
    with st.sidebar:
        st.header("Configuración LLM")
        with st.container():
            st.write("Proveedor")
            provider = st.selectbox("Proveedor", ["ollama", "gemini", "openai", "custom"], index=["ollama","gemini","openai","custom"].index(LLM_CONFIG.default_provider) if LLM_CONFIG.default_provider in ["ollama","gemini","openai","custom"] else 0, key='provider_select')
            api_key = st.text_input("API Key (si aplica)", value=LLM_CONFIG.gemini_api_key or LLM_CONFIG.openai_api_key or "", type="password")
            base_url = st.text_input("Base URL / Endpoint (si aplica)", value=getattr(LLM_CONFIG, 'gemini_base_url', '') or LLM_CONFIG.ollama_base_url or '')
            model_hint = st.text_input("Modelo (opcional)", value=LLM_CONFIG.default_model)
            discover = st.button("🔎 Buscar modelos disponibles")
        st.markdown("---")
        with st.expander("Ayuda rápida"):
            st.write("Sugerencias de prompts:")
            st.markdown("- 'Hola, necesito ayuda para una cita'\n- 'Tengo dolor de cabeza y fiebre desde ayer'\n- '¿Qué documentos necesito llevar?'")
            st.write("Consejos:")
            st.markdown("- Pega tu API key si usas Gemini / OpenAI.\n- Usa 'Buscar modelos' para detectar modelos disponibles.")
        if st.button("Reiniciar conversación"):
            reset_conversation()

    # Apply settings when discovery or model provided
    if discover:
        # apply temporary config
        LLM_CONFIG.default_provider = provider
        if api_key:
            LLM_CONFIG.gemini_api_key = api_key
            LLM_CONFIG.openai_api_key = api_key
        if base_url:
            LLM_CONFIG.gemini_base_url = base_url
            LLM_CONFIG.ollama_base_url = base_url
        # attempt discovery
        try:
            provider_inst = ProviderFactory.get_provider()
            models = provider_inst.list_models()
            if models:
                chosen = st.selectbox("Modelos detectados", models)
                LLM_CONFIG.default_model = chosen
                if provider == 'gemini':
                    LLM_CONFIG.gemini_model = chosen
                if provider == 'openai':
                    LLM_CONFIG.openai_model = chosen
                # remove existing manager so new provider is used
                if 'manager' in st.session_state:
                    del st.session_state['manager']
                st.success(f"Modelo seleccionado: {chosen}")

                # offer to save the selected model and creds immediately
                if st.button("💾 Guardar selección y credenciales en chatbot/.env"):
                    try:
                        env_path = Path(__file__).resolve().parent / '.env'
                        lines = []
                        if api_key:
                            lines.append(f"GEMINI_API_KEY={api_key}")
                            lines.append(f"OPENAI_API_KEY={api_key}")
                        if base_url:
                            lines.append(f"OLLAMA_BASE_URL={base_url}")
                        if LLM_CONFIG.default_model:
                            lines.append(f"DEFAULT_MODEL={LLM_CONFIG.default_model}")
                        env_text = "\n".join(lines) + "\n"
                        # write file with restrictive permissions
                        env_path.write_text(env_text, encoding='utf-8')
                        try:
                            import os
                            os.chmod(env_path, 0o600)
                        except Exception:
                            pass
                        st.success(f"Credenciales y modelo guardados en {env_path}")
                    except Exception as e:
                        st.error(f"No se pudo guardar .env: {e}")
            else:
                st.warning("No se encontraron modelos automáticamente. Introduce un nombre manualmente y aplica.")
        except Exception as e:
            st.error(f"Error inicializando proveedor: {e}")

            st.success(f"Credenciales guardadas en {env_path}")
        except Exception as e:
            st.error(f"No se pudo guardar .env: {e}")

    # Main layout: chat + state/prediction
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("Chat")
        for msg in st.session_state.messages_ui:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        prompt = st.chat_input("Escribe tu mensaje aquí...")
        if prompt:
            # append user message immediately to UI
            st.session_state.messages_ui.append({'role':'user','content':prompt})
            st.session_state.last_user_prompt = prompt
            st.session_state.last_error = None

            # ensure persistent manager in session
            try:
                if 'manager' not in st.session_state:
                    prov = ProviderFactory.get_provider()
                    st.session_state.manager = ConversationManager(prov)
                    st.session_state.predictor = Predictor()
                manager = st.session_state.manager
            except Exception as e:
                st.error(f"Error inicializando proveedor: {e}")
                st.session_state.last_error = str(e)
                manager = None

            # UI placeholders for status and assistant reply
            status_ph = st.empty()
            reply_ph = st.empty()

            def call_llm(p_text):
                try:
                    status_ph.info("Enviando al proveedor y generando respuesta...")
                    with st.spinner("LLM generando respuesta..."):
                        reply, ready_flag = manager.process_user_input(p_text)
                    # append assistant reply to UI
                    st.session_state.messages_ui.append({'role':'assistant','content':reply})
                    status_ph.success("Respuesta recibida")
                    st.session_state.last_error = None
                    return True
                except Exception as e:
                    err = str(e)
                    st.session_state.last_error = err
                    status_ph.error(f"Error: {err}")
                    return False

            # initial call
            success = False
            if manager is not None:
                success = call_llm(prompt)

            # If failed, show retry controls
            if not success:
                if st.button("Reintentar"):
                    if manager is None:
                        try:
                            prov = ProviderFactory.get_provider()
                            st.session_state.manager = ConversationManager(prov)
                            st.session_state.predictor = Predictor()
                            manager = st.session_state.manager
                        except Exception as e:
                            st.error(f"Error re-inicializando proveedor: {e}")
                            manager = None
                    if manager is not None:
                        call_llm(st.session_state.last_user_prompt)

            # scroll-like behavior: re-render messages list
            for msg in st.session_state.messages_ui:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])

    with col2:
        st.subheader("Estado paciente & Predicción")
        try:
            mgr = st.session_state.get('manager')
            if mgr is None:
                # lazy initialize persistent manager
                prov = ProviderFactory.get_provider()
                st.session_state.manager = ConversationManager(prov)
                st.session_state.predictor = Predictor()
                mgr = st.session_state.manager

            state_dump = mgr.get_current_state()
            missing = mgr.state.get_missing_critical_fields()
            render_patient_status(state_dump, missing)

            if mgr.state.is_ready_for_prediction():
                pred = st.session_state.predictor.predict(mgr.state)
                # colored risk badge
                color_class = 'risk-low'
                if pred.risk_level == 'ALTO':
                    color_class = 'risk-high'
                elif pred.risk_level == 'MEDIO':
                    color_class = 'risk-medium'
                st.markdown(f"<div class='risk-badge {color_class}'>Probabilidad de ausencia: {pred.probability:.2%} — {pred.risk_level}</div>", unsafe_allow_html=True)
                if pred.is_fallback:
                    st.caption("(Fallback usado — modelo ausente)")
        except Exception as e:
            st.error(f"Error mostrando estado/predicción: {e}")

if __name__ == '__main__':
    # ensure session defaults
    if 'messages_ui' not in st.session_state:
        st.session_state.messages_ui = [{'role':'assistant','content':'Hola. Soy el asistente del hospital. ¿En qué puedo ayudar?'}]
    main()