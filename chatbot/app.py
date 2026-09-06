"""
Módulo de la Interfaz de Usuario utilizando Streamlit.
Actúa exclusivamente como capa de presentación, delegando toda la lógica
al ConversationManager y al Predictor.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

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

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Page config + CSS (done once, at import time, not re-injected every rerun)
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Asistente de Admisión Hospitalaria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .chat-bubble { padding:10px; border-radius:10px; margin:8px 0; max-width:75%; }
    .chat-user { background:#e6f2ff; margin-left:auto; }
    .chat-assistant { background:#f1f8e9; margin-right:auto; }
    .chat-meta { font-size:0.8em; color:#666; margin-bottom:6px; }
    .risk-badge { padding:8px 12px; border-radius:6px; color:#fff; font-weight:600; display:inline-block;}
    .risk-high { background:#d32f2f; }
    .risk-medium { background:#f57c00; }
    .risk-low { background:#2e7d32; }
    .small-note { font-size:0.9em; color:#666; }
    </style>
    """,
    unsafe_allow_html=True,
)

PROVIDERS = ["ollama", "gemini", "openai", "custom"]


# --------------------------------------------------------------------------- #
# Cached / expensive-call helpers
# --------------------------------------------------------------------------- #

def _provider_cache_key() -> tuple:
    """Fingerprint of everything that changes which provider instance is valid.

    Used so we don't rebuild the provider (and its underlying client/session)
    on every single Streamlit rerun -- only when the config actually changes.
    """
    return (
        LLM_CONFIG.default_provider,
        LLM_CONFIG.default_model,
        getattr(LLM_CONFIG, "gemini_api_key", None),
        getattr(LLM_CONFIG, "openai_api_key", None),
        getattr(LLM_CONFIG, "ollama_base_url", None),
        getattr(LLM_CONFIG, "gemini_base_url", None),
        LLM_CONFIG.temperature,
        LLM_CONFIG.max_tokens,
        LLM_CONFIG.request_timeout,
    )


def get_manager() -> Optional[ConversationManager]:
    """Return a cached ConversationManager, rebuilding only when config changed.

    Avoids re-instantiating the LLM provider client on every rerun/keystroke,
    which is the single biggest avoidable cost in this app.
    """
    key = _provider_cache_key()
    if (
        "manager" not in st.session_state
        or st.session_state.get("_manager_key") != key
        or not hasattr(st.session_state.manager, "process_user_input")
    ):
        try:
            provider = ProviderFactory.get_provider()
            st.session_state.manager = ConversationManager(provider)
            st.session_state._manager_key = key
        except Exception as e:
            st.error(f"Error de configuración del proveedor LLM: {e}")
            return None
    return st.session_state.manager


@st.cache_data(ttl=300, show_spinner=False)
def discover_models_cached(provider_name: str, api_key: str, base_url: str) -> List[str]:
    """Cache model discovery for 5 minutes per (provider, key, url) combo.

    Model lists rarely change; this avoids re-hitting the provider's API
    every time the user touches an unrelated widget and triggers a rerun.
    """
    LLM_CONFIG.default_provider = provider_name
    if api_key:
        LLM_CONFIG.gemini_api_key = api_key
        LLM_CONFIG.openai_api_key = api_key
    if base_url:
        LLM_CONFIG.gemini_base_url = base_url
        LLM_CONFIG.ollama_base_url = base_url
    provider_inst = ProviderFactory.get_provider()
    return provider_inst.list_models() or []


def get_predictor() -> Predictor:
    if "predictor" not in st.session_state:
        st.session_state.predictor = Predictor()
    return st.session_state.predictor


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #

def initialize_session() -> None:
    """Initialize session_state defaults. Called once at the top of main()."""
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = [
            {
                "role": "assistant",
                "content": "Hola. Soy el asistente virtual del hospital. ¿En qué te puedo ayudar hoy?",
            }
        ]
    if "clinical_history_path" not in st.session_state:
        st.session_state.clinical_history_path = None
    if "discovered_models" not in st.session_state:
        st.session_state.discovered_models = []

    # Ensure manager/predictor exist without forcing a rebuild if valid.
    get_manager()
    get_predictor()


def reset_conversation() -> None:
    """Borra el estado actual para iniciar un nuevo flujo conversacional."""
    logger.info("Reiniciando la conversación a petición del usuario.")
    for key in ("manager", "_manager_key", "predictor", "messages_ui", "clinical_history_path", "discovered_models"):
        st.session_state.pop(key, None)
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> None:
    """Renderiza el panel lateral: proveedor LLM, credenciales y ayuda."""
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")

        provider = st.selectbox(
            "Proveedor",
            PROVIDERS,
            index=PROVIDERS.index(LLM_CONFIG.default_provider) if LLM_CONFIG.default_provider in PROVIDERS else 0,
            key="provider_select",
        )
        api_key = st.text_input(
            "API Key (si aplica)",
            value=LLM_CONFIG.gemini_api_key or LLM_CONFIG.openai_api_key or "",
            type="password",
        )
        base_url = st.text_input(
            "Base URL / Endpoint (si aplica)",
            value=getattr(LLM_CONFIG, "gemini_base_url", "") or LLM_CONFIG.ollama_base_url or "",
        )
        model_hint = st.text_input("Modelo (opcional)", value=LLM_CONFIG.default_model)

        LLM_CONFIG.temperature = st.slider(
            "Temperatura (Creatividad vs Precisión)",
            min_value=0.0, max_value=1.0, value=LLM_CONFIG.temperature, step=0.1,
            help="Mantenlo en 0.0 para maximizar la consistencia del JSON.",
        )
        LLM_CONFIG.max_tokens = st.number_input(
            "Max Tokens", min_value=64, max_value=4096, value=LLM_CONFIG.max_tokens, step=64
        )

        discover = st.button("🔎 Buscar modelos disponibles")
        if discover:
            try:
                with st.spinner("Buscando modelos..."):
                    models = discover_models_cached(provider, api_key, base_url)
                st.session_state.discovered_models = models
                if not models and model_hint:
                    st.warning("No se pudieron listar modelos automáticamente; se usará el nombre indicado.")
            except Exception as e:
                st.error(f"Error inicializando proveedor: {e}")

        # Persisted across reruns, unlike the original which vanished
        # as soon as `discover` went back to False on the next script run.
        if st.session_state.discovered_models:
            chosen = st.selectbox("Modelos detectados", st.session_state.discovered_models, key="chosen_model")
            if st.button("Usar este modelo"):
                _apply_llm_settings(provider, api_key, base_url, chosen)
        elif model_hint:
            if st.button("Aplicar configuración"):
                _apply_llm_settings(provider, api_key, base_url, model_hint)

        with st.expander("💾 Guardar credenciales"):
            _render_credential_persistence(provider, api_key, base_url)

        st.divider()
        with st.expander("Ayuda rápida"):
            st.markdown(
                "**Sugerencias de prompts:**\n"
                "- 'Hola, necesito ayuda para una cita'\n"
                "- 'Tengo dolor de cabeza y fiebre desde ayer'\n"
                "- '¿Qué documentos necesito llevar?'\n\n"
                "**Consejos:**\n"
                "- Pega tu API key si usas Gemini / OpenAI.\n"
                "- Usa 'Buscar modelos' para detectar modelos disponibles."
            )

        st.divider()
        if st.button("🔄 Nueva Conversación", use_container_width=True):
            reset_conversation()


def _apply_llm_settings(provider: str, api_key: str, base_url: str, model: str) -> None:
    """Apply chosen provider/model settings and invalidate the cached manager."""
    LLM_CONFIG.default_provider = provider
    LLM_CONFIG.default_model = model
    if api_key:
        LLM_CONFIG.gemini_api_key = api_key
        LLM_CONFIG.openai_api_key = api_key
    if base_url:
        LLM_CONFIG.gemini_base_url = base_url
        LLM_CONFIG.ollama_base_url = base_url
    if provider == "gemini":
        LLM_CONFIG.gemini_model = model
    if provider == "openai":
        LLM_CONFIG.openai_model = model
    st.success(f"Modelo aplicado: {model}")
    st.rerun()


def _render_credential_persistence(provider: str, api_key: str, base_url: str) -> None:
    """Plaintext .env save, plus optional encrypted save/load if `cryptography` is installed."""
    if st.button("Guardar en chatbot/.env"):
        try:
            env_path = Path(__file__).resolve().parent / ".env"
            lines = []
            if api_key:
                lines.append(f"GEMINI_API_KEY={api_key}")
                lines.append(f"OPENAI_API_KEY={api_key}")
            if base_url:
                lines.append(f"OLLAMA_BASE_URL={base_url}")
            if LLM_CONFIG.default_model:
                lines.append(f"DEFAULT_MODEL={LLM_CONFIG.default_model}")
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            try:
                os.chmod(env_path, 0o600)
            except Exception:
                pass
            st.success(f"Credenciales guardadas en {env_path}")
        except Exception as e:
            st.error(f"No se pudo guardar .env: {e}")

    try:
        from chatbot.utils.crypto import HAS_CRYPTO, encrypt_dict, decrypt_file
    except Exception:
        HAS_CRYPTO = False

    if not HAS_CRYPTO:
        st.caption("Instala 'cryptography' (pip install cryptography) para guardar credenciales encriptadas.")
        return

    passphrase = st.text_input("Passphrase para encriptar", type="password", key="enc_pass")
    passphrase2 = st.text_input("Confirmar passphrase", type="password", key="enc_pass2")
    if st.button("Encriptar y guardar .env.enc"):
        if not passphrase or passphrase != passphrase2:
            st.error("Las passphrases no coinciden o están vacías.")
        else:
            data = {}
            if api_key:
                data["GEMINI_API_KEY"] = api_key
                data["OPENAI_API_KEY"] = api_key
            if base_url:
                data["OLLAMA_BASE_URL"] = base_url
            if LLM_CONFIG.default_model:
                data["DEFAULT_MODEL"] = LLM_CONFIG.default_model
            try:
                enc = encrypt_dict(passphrase, data)
                env_enc_path = Path(__file__).resolve().parent / ".env.enc"
                env_enc_path.write_text(json.dumps(enc, ensure_ascii=False), encoding="utf-8")
                try:
                    os.chmod(env_enc_path, 0o600)
                except Exception:
                    pass
                st.success(f"Credenciales encriptadas guardadas en {env_enc_path}")
            except Exception as e:
                st.error(f"Fallo al encriptar: {e}")

    enc_path = Path(__file__).resolve().parent / ".env.enc"
    if enc_path.exists():
        dec_pass = st.text_input("Passphrase para desencriptar", type="password", key="dec_pass")
        if st.button("Cargar .env.enc"):
            try:
                data = decrypt_file(dec_pass, str(enc_path))
                if "GEMINI_API_KEY" in data:
                    LLM_CONFIG.gemini_api_key = data["GEMINI_API_KEY"]
                    LLM_CONFIG.openai_api_key = data["GEMINI_API_KEY"]
                if "OPENAI_API_KEY" in data:
                    LLM_CONFIG.openai_api_key = data["OPENAI_API_KEY"]
                if "OLLAMA_BASE_URL" in data:
                    LLM_CONFIG.ollama_base_url = data["OLLAMA_BASE_URL"]
                if "DEFAULT_MODEL" in data:
                    LLM_CONFIG.default_model = data["DEFAULT_MODEL"]
                st.success("Credenciales cargadas en la configuración de sesión")
                st.rerun()
            except Exception as e:
                st.error(f"Fallo al desencriptar: {e}")


# --------------------------------------------------------------------------- #
# Patient status panel
# --------------------------------------------------------------------------- #

def render_patient_status(state_dump: Dict[str, Any], missing_fields: list) -> None:
    st.subheader("📋 Estado del Paciente")

    if not state_dump:
        st.info("Aún no se ha recopilado información.")
        return

    extracted_data = {k: v for k, v in state_dump.items() if v.get("value") is not None}

    if extracted_data:
        # Single markdown write instead of one st.markdown call per field —
        # cheaper to render and avoids N separate DOM nodes.
        rows = []
        for key, data in extracted_data.items():
            value = data["value"]
            conf = data["confidence"]
            color = "green" if conf >= 0.8 else ("orange" if conf >= 0.5 else "red")
            rows.append(
                f"**{key}**: {value} "
                f"<span style='color:{color}; font-size:0.8em;'>(Confianza: {conf:.2f})</span>"
            )
        st.markdown("<br/>".join(rows), unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 Variables Pendientes")
    if missing_fields:
        st.markdown("\n".join(f"- `{field}`" for field in missing_fields))
    else:
        st.success("¡Información completada!")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    initialize_session()

    st.title("🏥 Asistente de Admisión Hospitalaria — Chatbot")
    st.caption("Interfaz para conversar con el asistente y ver el estado del paciente en tiempo real.")

    render_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Chat")

        # st.chat_message renders natively and incrementally — Streamlit only
        # diffs what changed, unlike the previous approach of rebuilding one
        # giant HTML string every rerun and force-scrolling it via an
        # injected <script> inside a components.html iframe.
        chat_box = st.container(height=500)
        with chat_box:
            for msg in st.session_state.messages_ui:
                avatar = "🩺" if msg["role"] == "assistant" else "🙋"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])

        user_text = st.chat_input("Escribe tu mensaje aquí...")

        if user_text and user_text.strip():
            st.session_state.messages_ui.append({"role": "user", "content": user_text})
            manager = get_manager()

            if manager is not None:
                try:
                    with st.spinner("El asistente está escribiendo..."):
                        reply, ready = manager.process_user_input(user_text)
                    st.session_state.messages_ui.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.messages_ui.append(
                        {"role": "assistant", "content": f"⚠️ Ocurrió un error: {e}"}
                    )
                    st.session_state.last_error = str(e)
            st.rerun()

    with col2:
        st.subheader("Estado paciente & Predicción")
        mgr = get_manager()
        if mgr is not None:
            try:
                state_dump = mgr.get_current_state()
                missing = mgr.state.get_missing_critical_fields()
                render_patient_status(state_dump, missing)

                if mgr.state.is_ready_for_prediction():
                    pred = get_predictor().predict(mgr.state)
                    color_class = "risk-low"
                    if pred.risk_level == "ALTO":
                        color_class = "risk-high"
                    elif pred.risk_level == "MEDIO":
                        color_class = "risk-medium"
                    st.markdown(
                        f"<div class='risk-badge {color_class}'>"
                        f"Probabilidad de ausencia: {pred.probability:.2%} — {pred.risk_level}</div>",
                        unsafe_allow_html=True,
                    )
                    if pred.is_fallback:
                        st.caption("(Fallback usado — modelo ausente)")

                # Export controls
                st.divider()
                st.subheader("📤 Exportar ficha / conversación")
                import io, csv
                # Prepare conversation JSON
                conv = {
                    "generated_at": datetime.utcnow().isoformat(),
                    "messages": st.session_state.get("messages_ui", []),
                    "patient_state": state_dump,
                }
                conv_json = json.dumps(conv, ensure_ascii=False, indent=2)
                st.download_button("Exportar ficha (JSON)", data=conv_json, file_name="ficha.json", mime="application/json")

                # Prepare conversation CSV
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow(["index", "role", "content"])
                for idx, m in enumerate(st.session_state.get("messages_ui", [])):
                    writer.writerow([idx, m.get("role"), m.get("content").replace("\n", " ")])
                st.download_button("Exportar conversación (CSV)", data=csv_buf.getvalue(), file_name="conversacion.csv", mime="text/csv")

                # Export patient state CSV (field, value, confidence)
                state_buf = io.StringIO()
                s_writer = csv.writer(state_buf)
                s_writer.writerow(["field", "value", "confidence"])
                if isinstance(state_dump, dict):
                    for k, v in state_dump.items():
                        val = v.get("value") if isinstance(v, dict) else str(v)
                        conf = v.get("confidence") if isinstance(v, dict) else ""
                        s_writer.writerow([k, val, conf])
                st.download_button("Exportar estado paciente (CSV)", data=state_buf.getvalue(), file_name="estado_paciente.csv", mime="text/csv")

                # Option to save clinical history text if available
                ch_path = st.session_state.get("clinical_history_path")
                if ch_path:
                    try:
                        ch_text = Path(ch_path).read_text(encoding="utf-8")
                        st.download_button("Descargar historia clínica (txt)", data=ch_text, file_name="historia_clinica.txt", mime="text/plain")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Error mostrando estado/predicción: {e}")


if __name__ == "__main__":
    main()