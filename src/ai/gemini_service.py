import logging
from typing import Generator, List, Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self, api_key: str, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._init_client()

    def _init_client(self):
        """Initialize Gemini client supporting google-genai or google-generativeai."""
        self.client_type = None
        
        # Try primary google.genai SDK
        try:
            from google import genai
            self.genai_client = genai.Client(api_key=self.api_key)
            self.client_type = "google_genai"
            return
        except (ImportError, Exception) as e:
            logger.debug(f"google.genai SDK init skipped: {e}")

        # Fallback to google.generativeai SDK
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self.legacy_model = genai_legacy.GenerativeModel(self.model_name)
            self.client_type = "google_generativeai"
            return
        except Exception as e:
            logger.error(f"google.generativeai SDK init failed: {e}")

        if not self.client_type:
            raise RuntimeError("Failed to initialize Google GenAI SDK. Please verify installed dependencies.")

    def stream_chat(
        self,
        history: List[Dict[str, Any]],
        user_text: str,
        image: Optional[Image.Image] = None,
        system_instruction: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generates streaming chat response for multimodal queries.
        
        Args:
            history: Previous chat messages list [{'role': 'user'/'assistant', 'content': '...'}]
            user_text: Current user question/prompt
            image: PIL Image object if attached to current turn
            system_instruction: System prompt/persona instruction
        """
        try:
            if self.client_type == "google_genai":
                yield from self._stream_genai_sdk(history, user_text, image, system_instruction)
            else:
                yield from self._stream_legacy_sdk(history, user_text, image, system_instruction)
        except Exception as e:
            err_str = str(e)
            if "ResourceExhausted" in err_str or "429" in err_str:
                yield "\n\n⚠️ **Rate Limit Exceeded (Free Tier)**: You have reached the Gemini API free tier rate limit (15 requests/min). Please wait a few seconds before sending another message."
            elif "API_KEY" in err_str or "API key" in err_str or "UNAUTHENTICATED" in err_str:
                yield "\n\n❌ **Authentication Error**: Invalid or missing Gemini API key. Please check your key in `.env` or the sidebar."
            else:
                yield f"\n\n❌ **API Error**: {err_str}"

    def _stream_genai_sdk(
        self,
        history: List[Dict[str, Any]],
        user_text: str,
        image: Optional[Image.Image],
        system_instruction: Optional[str]
    ) -> Generator[str, None, None]:
        from google import genai
        from google.genai import types

        # Build content payload
        contents = []

        # Include historical conversation text for multi-turn context
        for msg in history:
            role = msg.get("role")
            content_text = msg.get("content", "")
            if role and content_text:
                prefix = "User: " if role == "user" else "Assistant: "
                contents.append(f"{prefix}{content_text}")

        # Build current turn content (Image + User prompt)
        current_parts = []
        if image:
            current_parts.append(image)
        if user_text:
            current_parts.append(user_text)

        contents.append(current_parts)

        # Config with system instruction if present
        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            )

        response = self.genai_client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _stream_legacy_sdk(
        self,
        history: List[Dict[str, Any]],
        user_text: str,
        image: Optional[Image.Image],
        system_instruction: Optional[str]
    ) -> Generator[str, None, None]:
        import google.generativeai as genai_legacy

        model = genai_legacy.GenerativeModel(
            self.model_name,
            system_instruction=system_instruction
        )

        contents = []
        if image:
            contents.append(image)
        contents.append(user_text)

        response = model.generate_content(contents, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
