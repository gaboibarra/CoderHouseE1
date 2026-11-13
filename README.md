# Unified Async LLM Client

**Pre-entrega 1 — Cliente de LLM robusto y asíncrono**

Python 3.12 | OpenAI | Anthropic | Pydantic | asyncio | Streaming

## Descripción

Este proyecto implementa un cliente unificado para trabajar con dos proveedores de modelos de lenguaje (LLM): **OpenAI y Anthropic**.

Permite realizar consultas asíncronas, recibir respuestas completas y utilizar streaming para mostrar fragmentos de texto a medida que llegan.

Ambos proveedores se implementan mediante una interfaz común, por lo que el código principal puede utilizar cualquiera de ellos sin modificar la lógica de comunicación.

## Funcionalidades

- Intercambiabilidad entre OpenAI y Anthropic.
- Programación asíncrona utilizando `async/await`.
- Streaming mediante generadores asíncronos (`async for` y `yield`).
- Validación de mensajes y configuración mediante Pydantic.
- Manejo controlado de errores de autenticación, red, timeout y rate limiting.
- Reintentos automáticos del SDK para determinadas fallas transitorias.
- Pruebas automatizadas con pytest.

## Estructura del proyecto

```text
unified-async-llm-client/
├── llm_client/
│   ├── __init__.py
│   ├── schemas.py
│   ├── base.py
│   ├── providers.py
│   ├── manager.py
│   └── errors.py
├── tests/
│   └── test_schemas.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore

```

## Arquitectura

La implementación utiliza una clase base abstracta llamada `BaseLLMClient`, que establece los métodos que todos los proveedores deben implementar.

- **BaseLLMClient:** define los contratos `generate()`, `stream()` y `close()`.
- **OpenAIClient:** utiliza el SDK oficial `AsyncOpenAI`.
- **AnthropicClient:** utiliza el SDK oficial `AsyncAnthropic`.
- **AsyncLLMManager:** selecciona el proveedor mediante configuración.
- **schemas.py:** contiene los modelos Pydantic para validar entradas, configuración y respuestas.
- **errors.py:** define errores propios para unificar el manejo de excepciones.

## Requisitos

- Python 3.12. (En este caso se uso la version 3.14)
- Acceso a la API de OpenAI.
- Acceso a la API de Anthropic.
- Claves API válidas para ambos proveedores.

## Instalación

### 1. Crear el entorno virtual

Desde la carpeta del proyecto:

```powershell
py -3.14 -m venv .venv
```

Activar el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar las dependencias

```powershell
python -m pip install -r requirements.txt
```

Las dependencias principales son `openai`, `anthropic`, `pydantic` y `python-dotenv`. También se utiliza `pytest` para realizar pruebas.

## Configuración de las API keys

Crear un archivo `.env` a partir de `.env.example`:

```powershell
Copy-Item .env.example .env
```

Completar las variables:

```dotenv
OPENAI_API_KEY=tu_clave_openai
ANTHROPIC_API_KEY=tu_clave_anthropic

OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-haiku-4-5
```

Los modelos pueden modificarse según los disponibles en cada cuenta.

**Importante:** nunca publicar el archivo `.env` en GitHub. Las claves API son privadas.

## Ejecución del programa

Ejecutar:

```powershell
python main.py
```

El programa realiza la pregunta:

**¿Qué es la entropía?**

Primero ejecuta OpenAI y después Anthropic.

Para cada proveedor realiza dos modalidades de consulta.

### Generación normal

```python
response = await manager.generate(messages)
```

Devuelve una respuesta estructurada mediante `ModelResponse`, incluyendo el texto generado, el proveedor, el modelo y los datos disponibles de uso de tokens.

<img width="637" height="80" alt="image" src="https://github.com/user-attachments/assets/eab162da-be89-40ea-9397-0ebde2b5f301" />

<img width="641" height="164" alt="image" src="https://github.com/user-attachments/assets/c7838b16-22c1-4600-80ed-bb21dab2b973" />

### Generación mediante streaming

```python
async for fragmento in manager.stream(messages):
    print(fragmento, end="", flush=True)
```

Entrega los fragmentos a medida que llegan desde la API, sin esperar a que finalice toda la generación.

<img width="635" height="30" alt="image" src="https://github.com/user-attachments/assets/93a4c532-c7e0-4baf-b894-5ad3ea63d8e0" />

<img width="632" height="104" alt="image" src="https://github.com/user-attachments/assets/cf2fc673-bf96-4bb9-8dbf-d6e9f22a17cf" />


## Resultados de ejecución

Ambos proveedores se ejecutaron correctamente durante las pruebas manuales.

| Funcionalidad | OpenAI | Anthropic |
|---|---|---|
| Conexión con la API | Correcta | Correcta |
| Generación normal | Correcta | Correcta |
| Streaming | Correcto | Correcto |
| Conteo de tokens | Correcto | Correcto |

Las respuestas varían entre ejecuciones. El número máximo de tokens configurado puede provocar que respuestas extensas se corten al alcanzar ese límite.

## Pruebas automatizadas

Ejecutar:

```powershell
python -m pytest -q
```

Resultado obtenido:

```text
6 passed in 2.47s
```

<img width="638" height="44" alt="image" src="https://github.com/user-attachments/assets/b81e6588-ccaf-47df-896f-0cf48e8083db" />

Las seis pruebas automatizadas se completaron correctamente.

Estas pruebas verifican casos de validación. Las llamadas reales a las APIs se comprobaron mediante el script `main.py`.

Para consultar individualmente los tests:

```powershell
python -m pytest -v
```
<img width="638" height="151" alt="image" src="https://github.com/user-attachments/assets/4590ed03-293f-437d-9680-16ff6815e4c8" />

## Manejo de errores

La implementación contempla errores como:

- Clave API incorrecta.
- Límite de solicitudes o cuota.
- Problemas de conexión.
- Timeout.
- Otros errores de las APIs.

En generación normal, los errores se representan mediante una respuesta estructurada con `success=False`.

En streaming, se utiliza la excepción personalizada `LLMClientError`, que puede capturarse desde el programa principal.

Los SDKs utilizan hasta dos reintentos configurados para determinadas fallas transitorias.

## Limitaciones y posibles mejoras

La implementación actual trabaja con mensajes de texto y tres roles: `system`, `user` y `assistant`.

Como futuras mejoras se propone:

- Incorporar pruebas con mocks para simular errores y respuestas de ambos SDKs.
- Detectar explícitamente respuestas truncadas.
- Agregar otros proveedores.
- Incorporar métricas de latencia y consumo.
- Ampliar el soporte a herramientas y respuestas estructuradas.

## Referencias

- [SDK oficial de OpenAI](https://github.com/openai/openai-python)
- [SDK oficial de Anthropic](https://github.com/anthropics/anthropic-sdk-python)
- [Documentación de Pydantic](https://docs.pydantic.dev/latest/)
- [Documentación de asyncio](https://docs.python.org/3/library/asyncio.html)

## Seguridad

Las credenciales se almacenan mediante variables de entorno y el archivo `.env` se excluye del control de versiones mediante `.gitignore`.

No se incluyen claves API reales en el repositorio.
