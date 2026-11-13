# ABC define una clase abstracta; abstractmethod obliga a implementar métodos en las clases hijas.
from abc import ABC, abstractmethod
# AsyncIterator[str] representa una secuencia asíncrona de fragmentos de texto.
from collections.abc import AsyncIterator

from .schemas import ChatMessage, ModelConfig, ModelResponse


# Contrato de comportamiento: todos los proveedores deben ofrecer estos métodos.
class BaseLLMClient(ABC):
    # Guardamos la configuración validada como atributo del cliente.
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    # Una corrutina obtiene la respuesta completa y estandarizada.
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Generar respuesta completa; errores de API se devuelven estructurados."""

    @abstractmethod
    # Un generador asíncrono entrega texto progresivamente.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Emitir fragmentos; fallas se reportan mediante LLMClientError."""

    @abstractmethod
    # Cierra conexiones HTTP y libera recursos del SDK.
    async def close(self) -> None:
        """Cerrar recursos de red del SDK."""
