import base64
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
# Load environment variables from .env file
load_dotenv()


def generate(
    user_prompt: str,
    temperature: int = 0.5,
    ):
    """
        Generate a response from the Gemini model.

        Args:
            user_prompt: the prompt to generate a response from the Gemini model.
            temperature: the temperature setting passed to the Gemini API

        Returns:
            The response from the Gemini model.
    """

    # Initialize the client
    client = genai.Client(
        api_key=os.environ.get("ADI_GOOGLE_AI_STUDIO_API_KEY"),
    )

    model = "gemini-2.5-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    response = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text

    return response


if __name__ == "__main__":
    generate()
