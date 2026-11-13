"""Demostración de intercambiabilidad: prueba OpenAI y Anthropic en una ejecución."""

# asyncio permite ejecutar las funciones async dentro de un event loop.
import asyncio
# os.getenv lee variables configuradas en el archivo .env.
import os

# Carga el archivo .env para no colocar API keys en el código fuente.
from dotenv import load_dotenv
# Permite distinguir errores de validación de configuración.
from pydantic import ValidationError

# Importamos la interfaz pública del paquete; main no necesita importar los SDK.
from llm_client import AsyncLLMManager, ChatMessage, LLMClientError, ModelConfig


# La misma función sirve para los DOS proveedores: eso demuestra intercambiabilidad.
async def probar_proveedor(nombre: str, modelo: str) -> None:
    print(f"\n{'=' * 55}\nPROVEEDOR: {nombre.upper()} | MODELO: {modelo}\n{'=' * 55}")

    # Construye la configuración compartida, validada por Pydantic.
    config = ModelConfig(provider=nombre, model=modelo, max_tokens=150)

    # Enviamos exactamente la misma conversación a ambos clientes.
    messages = [
        ChatMessage(role="system", content="Respondé en español y de forma breve."),
        ChatMessage(role="user", content="¿Qué es la entropía?"),
    ]

    # El manager construye OpenAIClient o AnthropicClient según config.provider.
    # async with asegura el cierre de la conexión aun si ocurre un problema.
    async with AsyncLLMManager(config) as manager:
        print("\n--- Generación normal ---")
        # await espera la respuesta sin bloquear el event loop.
        respuesta = await manager.generate(messages)
        if respuesta.success:
            print(respuesta.content)
            print(f"Tokens: entrada={respuesta.input_tokens}, salida={respuesta.output_tokens}")
        else:
            # Errores esperables: cuota, credenciales, red, timeout, etc.
            print(f"Error controlado [{respuesta.error_code}]: {respuesta.error_message}")
            # No intentamos streaming si la generación ya falló.
            return

        print("\n--- Streaming ---")
        try:
            # El generador async produce fragmentos de texto conforme llegan.
            async for fragmento in manager.stream(messages):
                print(fragmento, end="", flush=True)
            print()  # Salto de línea después del último fragmento.
        except LLMClientError as error:
            # En streaming las fallas se señalan con una excepción común.
            print(f"\nError controlado [{error.code}]: {error.message}")


async def main() -> None:
    # Cargamos las claves OPENAI_API_KEY y ANTHROPIC_API_KEY desde .env.
    load_dotenv()

    # Lista de proveedores a probar. Son configurables sin modificar el cliente.
    proveedores = [
        ("openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        ("anthropic", os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")),
    ]

    # Ambos proveedores usan exactamente la misma función probar_proveedor.
    for nombre, modelo in proveedores:
        try:
            await probar_proveedor(nombre, modelo)
        except (ValueError, ValidationError) as error:
            # Si falta una API key, informamos y seguimos con el otro proveedor.
            print(f"Error de configuración de {nombre}: {error}")


# Evita que main() se ejecute automáticamente al importar este archivo en un test.
if __name__ == "__main__":
    # Inicia el event loop y ejecuta nuestra función principal asíncrona.
    asyncio.run(main())
