# Permite leer las claves API guardadas como variables de entorno.
import os
from collections.abc import AsyncIterator

from .base import BaseLLMClient
from .providers import AnthropicClient, OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


# Es la fachada: el resto del programa no conoce los detalles de cada SDK.
class AsyncLLMManager:
    # Elegimos proveedor usando la configuración y creamos su cliente.
    def __init__(self, config: ModelConfig):
        self.config = config
        # Si el proveedor solicitado es OpenAI...
        if config.provider == Provider.OPENAI:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Falta OPENAI_API_KEY en el archivo .env")
            self.client: BaseLLMClient = OpenAIClient(config, key)
        # Si el proveedor solicitado es Anthropic...
        elif config.provider == Provider.ANTHROPIC:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Falta ANTHROPIC_API_KEY en el archivo .env")
            self.client = AnthropicClient(config, key)
        # Un proveedor desconocido se rechaza con un error claro.
        else:
            raise ValueError(f"Proveedor desconocido: {config.provider}")

    # Delegamos la solicitud completa a la implementación seleccionada.
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    # Reenviamos cada fragmento que llega desde el proveedor.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async for fragment in self.client.stream(messages):
            yield fragment

    # El manager delega también el cierre de conexiones.
    async def close(self) -> None:
        await self.client.close()

    # Permite usar async with y obtener el manager dentro del bloque.
    async def __aenter__(self):
        return self

    # Se ejecuta al salir de async with, incluso si ocurrió una excepción.
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()
