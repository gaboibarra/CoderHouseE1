# Enum permite definir valores permitidos con nombres claros.
from enum import Enum

# BaseModel valida datos; Field configura restricciones; model_validator comprueba reglas entre campos.
from pydantic import BaseModel, Field, model_validator


# Solo admitimos estos dos proveedores; heredar str permite trabajar con sus nombres.
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Roles permitidos en una conversación.
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# Cada mensaje debe tener un rol válido y texto no vacío.
class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


# Configuración única independientemente del proveedor.
class ModelConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    timeout_seconds: float = Field(default=30, gt=0, le=300)

    # Validación adicional después de verificar los campos individuales.
    @model_validator(mode="after")
    # En Anthropic, el parámetro temperature no debe superar 1.
    def check_anthropic_temperature(self):
        if self.provider == Provider.ANTHROPIC and self.temperature is not None and self.temperature > 1:
            raise ValueError("Anthropic requiere temperature entre 0 y 1")
        return self


# Formato común tanto para resultados correctos como para errores controlados.
class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str = ""
    success: bool
    error_code: str | None = None
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
