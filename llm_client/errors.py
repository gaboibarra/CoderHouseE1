# Error propio de la aplicación; oculta las distintas excepciones de cada SDK.
class LLMClientError(Exception):
    """Error controlado, independiente del proveedor."""

    # code es legible por software y message por personas.
    def __init__(self, code: str, message: str):
        # Inicializa el mensaje de la excepción estándar de Python.
        super().__init__(message)
        self.code = code
        self.message = message
