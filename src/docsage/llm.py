"""Stage 6a: the language model client.

Talks to any OpenAI-compatible chat API: Groq, OpenAI, Together, a local
Ollama or vLLM server. Errors are translated into LLMError with a message a
user can act on, so the API layer does not need to know about the SDK.
"""

from typing import Protocol

import openai


class LLMError(Exception):
	def __init__(self, message: str, status_code: int = 502):
		super().__init__(message)
		self.status_code = status_code


class LLM(Protocol):
	model: str

	def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:
	def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 30.0):
		self.model = model
		self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

	def complete(self, system: str, user: str) -> str:
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				temperature=0,
				messages=[
					{"role": "system", "content": system},
					{"role": "user", "content": user},
				],
			)
		except openai.AuthenticationError as e:
			raise LLMError("The LLM API key was rejected. Check LLM_API_KEY.", 502) from e
		except openai.NotFoundError as e:
			raise LLMError(f"Model '{self.model}' was not found at {self.client.base_url}.", 502) from e
		except openai.RateLimitError as e:
			raise LLMError("The LLM provider is rate limiting requests. Try again shortly.", 429) from e
		except openai.APIConnectionError as e:
			raise LLMError(f"Cannot reach the LLM at {self.client.base_url}.", 503) from e
		except openai.APIStatusError as e:
			raise LLMError(f"The LLM provider returned an error ({e.status_code}).", 502) from e

		return response.choices[0].message.content or ""
