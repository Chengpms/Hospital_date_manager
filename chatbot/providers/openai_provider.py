"""
Proveedor para la API de OpenAI (Chat completions). Requiere la librer\u00eda openai.
Este proveedor intenta crear respuestas en formato JSON y limpiar code fences.
"""

import logging
from typing import List, Dict, Any

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        try:
            import openai
        except ModuleNotFoundError as e:
            raise RuntimeError("El paquete 'openai' no est\u00e1 instalado. Instala con 'pip install openai'.") from e
        self.openai = openai
        if getattr(LLM_CONFIG, 'openai_api_key', None):
            self.openai.api_key = LLM_CONFIG.openai_api_key
        self.model = getattr(LLM_CONFIG, 'openai_model', None) or LLM_CONFIG.default_model
        self.timeout = LLM_CONFIG.request_timeout
        logger.info(f"OpenAIProvider inicializado (Modelo: {self.model})")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        # Convertimos al formato que espera OpenAI
        chat_messages = []
        for m in messages:
            role = 'assistant' if m.get('role') == 'assistant' else 'user'
            chat_messages.append({'role': role, 'content': m.get('content')})

        try:
            resp = self.openai.ChatCompletion.create(
                model=self.model,
                messages=chat_messages,
                temperature=LLM_CONFIG.temperature,
                max_tokens=LLM_CONFIG.max_tokens,
            )
            text = resp.choices[0].message.content
            return self._clean_json_response(text)
        except Exception as e:
            logger.exception('Error llamando a OpenAI API')
            raise RuntimeError(str(e)) from e
