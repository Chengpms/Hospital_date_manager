"""
Implementación del proveedor LLM utilizando la API oficial de Google Gemini.
Garantiza el aislamiento de la lógica de comunicación externa.
"""

import logging
from google import genai
from google.genai import types
from typing import List, Dict, Any

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)

class GeminiProvider(BaseProvider):
    """
    Proveedor para comunicarse con la API de Google Gemini.
    Su única responsabilidad es enviar y recibir mensajes.
    """

    def __init__(self) -> None:
        """
        Inicializa el proveedor configurando la API Key.
        Valida que la credencial exista antes de instanciar el modelo.
        """
        api_key = LLM_CONFIG.gemini_api_key
        if not api_key:
            error_msg = "GEMINI_API_KEY no está configurada en las variables de entorno."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.model_name = LLM_CONFIG.gemini_model or "gemini-2.0-flash"
        self.client = genai.Client(api_key=api_key)
        self.generation_config = types.GenerateContentConfig(
            temperature=LLM_CONFIG.temperature,
            response_mime_type="application/json",
        )
        logger.info(f"GeminiProvider inicializado con el modelo: {self.model_name}")

    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convierte el formato estándar de mensajes al formato requerido por Gemini.
        """
        gemini_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            gemini_role = "model" if role == "assistant" else "user"
            gemini_messages.append({
                "role": gemini_role,
                "parts": [{"text": msg["content"]}]
            })
        return gemini_messages

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Envía la petición a Gemini.
        
        Args:
            messages: Historial y contexto estructurado.
            
        Returns:
            str: Respuesta del LLM limpia y en formato JSON.
            
        Raises:
            RuntimeError: Si ocurre un error en la comunicación con la API.
        """
        try:
            logger.debug("Enviando petición a Gemini API")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self._convert_messages_format(messages),
                config=self.generation_config,
            )
            text = response.text if hasattr(response, "text") else str(response)
            return self._clean_json_response(text)
        except Exception as e:
            error_msg = f"Error en la comunicación con Gemini API: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e