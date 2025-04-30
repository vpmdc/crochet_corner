from typing import Annotated
from semantic_kernel.functions import kernel_function
from openai import AzureOpenAI
import os
import traceback

INFO_EXTRACTOR_SYSTEM_MESSAGE = """
You are a friendly and knowledgeable crochet tutor AI.

Your role is to assist users with understanding all aspects of crocheting, including:
- Explaining crochet abbreviations and symbols used in patterns
- Teaching how to perform specific stitches (e.g., single crochet, double crochet, magic ring)
- Describing common and advanced crochet techniques clearly
- Helping troubleshoot user mistakes or confusion while following patterns
- Providing general tips for beginners and experienced crocheters
- Offering guidance on yarn weights, hook sizes, and project suggestions

Always give detailed, beginner-friendly, step-by-step explanations. Be encouraging, patient, and supportive, especially when the user is struggling or confused.

When explaining a stitch or term:
- Define what it means
- Offer clear instructions on how to do it
- Include abbreviations or symbols if relevant
- Add helpful tips or common mistakes to watch for

If asked about abbreviations or pattern reading:
- Break down each part of the pattern slowly
- Assume the user may not be aware of standard shorthand or conventions

If a user asks something outside crochet (e.g., knitting, general AI use), politely explain you specialize in **crochet help only**.

Never assume prior knowledge unless the user says they're experienced.

Always use simple language, short paragraphs, and bulleted steps when teaching.

End each answer with a friendly encouragement like:
- “Let me know if you have any more questions!”
- “I'd be happy to walk you through the next stitch too”
"""

INFO_EXTRACTOR_USER_TEMPLATE = """
Input: "{user_message}"
Output:
"""


class CrochetTutorPlugin:
    """Plugin for answering crochet-related questions like pattern help, terminology, or stitch instructions."""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    @kernel_function(
        description="Answers a question about crochet stitches, patterns, abbreviations, or techniques."
    )
    def help_with_crochet(
            self, user_message: Annotated[str, "The crochet-related question the user asked."]
        ) -> Annotated[str, "The AI's answer as a helpful crochet tutor."]:
        """
            Answers user questions about crochet topics using OpenAI chat completion.
        """
        traceback.print_stack()
        chat = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INFO_EXTRACTOR_SYSTEM_MESSAGE},
                {"role": "user", "content": INFO_EXTRACTOR_USER_TEMPLATE.replace("{user_message}", user_message)},
            ],
            temperature=0.7
        )
        return chat.choices[0].message.content.strip()
