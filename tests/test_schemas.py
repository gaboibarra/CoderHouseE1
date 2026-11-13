# Framework para ejecutar tests de forma automática.
import pytest
# Permite verificar que se detectan entradas inválidas.
from pydantic import ValidationError

from llm_client.schemas import ChatMessage, ModelConfig


# Comprobar que se acepta un mensaje bien construido.
def test_valid_message():
    msg = ChatMessage(role="user", content="Hola")
    assert msg.role.value == "user"


# Comprobar que no se aceptan roles desconocidos.
def test_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="unknown", content="Hola")


# Ejecuta la misma prueba con dos valores de temperatura inválidos.
@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_invalid_temperature(temperature):
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="test", temperature=temperature)


# Verifica la restricción específica de Anthropic.
def test_anthropic_temperature_limit():
    with pytest.raises(ValidationError):
        ModelConfig(provider="anthropic", model="test", temperature=1.5)


# Un máximo de tokens igual a cero debe rechazarse.
def test_invalid_max_tokens():
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="test", max_tokens=0)
