"""Mock backend for testing.

Returns the prompt as the agent's reply, or raises an error for "<FAIL>" prompt.
"""


class MockBackendError(Exception):
    """Error raised by MockBackend for invalid inputs."""

    pass


class MockBackend:
    """Mock backend that echoes prompts back as responses.

    Contract:
      async run(prompt: str) -> str

    Behavior:
      - Returns the prompt string as the response
      - Raises MockBackendError if prompt is "<FAIL>"
    """

    def __init__(self, endpoint: str = ""):
        """Initialize mock backend.

        Args:
            endpoint: Ignored, included for interface compatibility.
        """
        self.endpoint = endpoint

    async def run(self, prompt: str) -> str:
        """Execute a prompt and return the response.

        Args:
            prompt: The prompt to process.

        Returns:
            The prompt string echoed back.

        Raises:
            MockBackendError: If prompt is "<FAIL>".
        """
        if prompt == "<FAIL>":
            raise MockBackendError("Simulated backend failure")
        return prompt
