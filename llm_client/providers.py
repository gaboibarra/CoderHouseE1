# Tipo del retorno de los generadores asíncronos.
from collections.abc import AsyncIterator

# Importamos el módulo para reconocer sus tipos de excepciones.
import anthropic
# Importamos el módulo para reconocer sus tipos de excepciones.
import openai
# Cliente oficial asíncrono de Anthropic.
from anthropic import AsyncAnthropic
# Cliente oficial asíncrono de OpenAI.
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .errors import LLMClientError
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role


# Valida reglas de negocio: debe existir al menos un mensaje de usuario.
def _ensure_messages(messages: list[ChatMessage]) -> None:
    if not messages or not any(m.role == Role.USER for m in messages):
        raise ValueError("Debes incluir al menos un mensaje con role='user'")


# Traduce errores específicos de OpenAI a una excepción unificada.
# Se chequean primero clases de error específicas y luego las generales.
def _map_openai_error(exc: Exception) -> LLMClientError:
    if isinstance(exc, openai.AuthenticationError):
        return LLMClientError("authentication", "API key de OpenAI inválida")
    if isinstance(exc, openai.RateLimitError):
        return LLMClientError("rate_limit", "Límite de solicitudes o cuota de OpenAI")
    if isinstance(exc, openai.APITimeoutError):
        return LLMClientError("timeout", "Tiempo de espera agotado en OpenAI")
    if isinstance(exc, openai.APIConnectionError):
        return LLMClientError("network", "Error de conexión con OpenAI")
    if isinstance(exc, openai.APIStatusError):
        return LLMClientError("api_error", f"OpenAI devolvió HTTP {exc.status_code}")
    return LLMClientError("api_error", "Error inesperado del SDK OpenAI")


# Hace la misma traducción para errores de Anthropic.
def _map_anthropic_error(exc: Exception) -> LLMClientError:
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMClientError("authentication", "API key de Anthropic inválida")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMClientError("rate_limit", "Límite de solicitudes o cuota de Anthropic")
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMClientError("timeout", "Tiempo de espera agotado en Anthropic")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMClientError("network", "Error de conexión con Anthropic")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMClientError("api_error", f"Anthropic devolvió HTTP {exc.status_code}")
    return LLMClientError("api_error", "Error inesperado del SDK Anthropic")


# Implementación concreta de la interfaz común para OpenAI.
class OpenAIClient(BaseLLMClient):
    # Llama al constructor base y configura el SDK asíncrono con timeout y reintentos.
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=2,
        )

    # Prepara los argumentos (payload) con el formato que espera este proveedor.
    def _kwargs(self, messages: list[ChatMessage]) -> dict:
        _ensure_messages(messages)
        payload = {
            "model": self.config.model,
            # model_dump() convierte cada objeto Pydantic en un diccionario para el SDK.
            "messages": [m.model_dump() for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload

    # Modo normal: await espera una respuesta completa sin bloquear el event loop.
    # Si el SDK falla, convertimos la excepción en un ModelResponse de error.
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        payload = self._kwargs(messages)
        try:
            # **payload expande un diccionario en argumentos con nombre.
            result = await self.client.chat.completions.create(**payload)
            content = result.choices[0].message.content or ""
            usage = result.usage
            return ModelResponse(
                provider=Provider.OPENAI, model=self.config.model,
                content=content, success=True,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            )
        except openai.APIError as exc:
            error = _map_openai_error(exc)
            return ModelResponse(
                provider=Provider.OPENAI, model=self.config.model, success=False,
                error_code=error.code, error_message=error.message,
            )

    # Modo streaming: async for consume fragmentos y yield los entrega al llamador.
    # Si falla durante el streaming, lanzamos LLMClientError para no mezclar errores con texto.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        payload = self._kwargs(messages)
        try:
            # stream=True pide recibir fragmentos a medida que llegan.
            result = await self.client.chat.completions.create(**payload, stream=True)
            async for chunk in result:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        # yield produce el fragmento de OpenAI sin finalizar el generador.
                        yield delta
        except openai.APIError as exc:
            raise _map_openai_error(exc) from exc

    # Cierra conexiones de red del cliente asíncrono.
    async def close(self) -> None:
        await self.client.close()


# Implementación concreta de la interfaz común para Anthropic.
class AnthropicClient(BaseLLMClient):
    # Llama al constructor base y configura el SDK asíncrono con timeout y reintentos.
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=2,
        )

    # Prepara los argumentos (payload) con el formato que espera este proveedor.
    def _kwargs(self, messages: list[ChatMessage]) -> dict:
        _ensure_messages(messages)
        # Anthropic recibe el system prompt por separado.
        # Anthropic recibe los mensajes system en el campo system por separado.
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        # El campo messages solo lleva mensajes user/assistant.
        conversation = [m.model_dump() for m in messages if m.role != Role.SYSTEM]
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": conversation,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload

    # Modo normal: await espera una respuesta completa sin bloquear el event loop.
    # Si el SDK falla, convertimos la excepción en un ModelResponse de error.
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        payload = self._kwargs(messages)
        try:
            # Llamada asíncrona a la API Messages de Anthropic.
            result = await self.client.messages.create(**payload)
            content = "".join(block.text for block in result.content if block.type == "text")
            return ModelResponse(
                provider=Provider.ANTHROPIC, model=self.config.model,
                content=content, success=True,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
        except anthropic.APIError as exc:
            error = _map_anthropic_error(exc)
            return ModelResponse(
                provider=Provider.ANTHROPIC, model=self.config.model, success=False,
                error_code=error.code, error_message=error.message,
            )

    # Modo streaming: async for consume fragmentos y yield los entrega al llamador.
    # Si falla durante el streaming, lanzamos LLMClientError para no mezclar errores con texto.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        payload = self._kwargs(messages)
        try:
            # async with maneja apertura y cierre del stream de Anthropic.
            async with self.client.messages.stream(**payload) as result:
                # Recorremos fragmentos de texto provenientes del SDK.
                async for fragment in result.text_stream:
                    # yield produce el fragmento sin finalizar el generador.
                    yield fragment
        except anthropic.APIError as exc:
            raise _map_anthropic_error(exc) from exc

    # Cierra conexiones de red del cliente asíncrono.
    async def close(self) -> None:
        await self.client.close()
