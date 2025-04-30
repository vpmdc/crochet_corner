import os
from typing import Annotated, List
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from openai import AzureOpenAI
from semantic_kernel.functions import kernel_function
from azure.core.credentials import  AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
import os

INFO_EXTRACTOR_SYSTEM_MESSAGE = """
AI Assistant Parsing Prompt
You are an AI assistant that helps users search for crochet patterns.
Your job is to extract two things from the user's input:
The search query (the actual pattern or description the user is interested in)
The number of results they want (if specified; otherwise, use 10)

Instructions:

If the user mentions a number (e.g., "Show me 5 tote bags"), use that as the number of results.
If no number is mentioned, default to 10. Remove any command phrases like "show me", "find", "list", etc. from the search query.

Output your answer in this JSON format:

{
  "semantic_query": "<the search query>",
  "num_patterns": <number>
}

"""

INFO_EXTRACTOR_USER_MESSAGE = """
Input: "{user_input}"
Output:
"""

class PatternSearchPlugin:
    @kernel_function
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.search_client = SearchClient(
            endpoint=os.getenv("SEARCH_ENDPOINT"),
            index_name="search-pattern-detail-index",
            credential=AzureKeyCredential(os.getenv("SEARCH_ADMIN_KEY"))
        )

    @kernel_function(
        name="parse_pattern_search_user_input",
        description="Parses user input to extract a semantic query for pattern search."
    )
    def parse_pattern_finding_input(self, user_input: dict) -> dict:
        """
        Extracts the semantic query from user input for pattern search.
        Removes common command phrases and numbers.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with your deployed model
                messages=[
                    {"role": "system", "content": INFO_EXTRACTOR_SYSTEM_MESSAGE},
                    {"role": "user", "content": INFO_EXTRACTOR_USER_MESSAGE.replace('{user_input}', user_input)}
                ],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to parse user input"
    
    @kernel_function(
        name="get_matching_patterns",
        description="Finds top matching patterns for a given semantic query. Maximum 15 results."
    )
    def execute_pattern_query_search(
            self,
            semantic_query: Annotated[str, "A semantic description of the desired crochet pattern."],
            num_patterns: Annotated[int, "Number of patterns to return, up to a maximum of 15."]
        ) -> Annotated[str, "A formatted string containing the matched crochet patterns."]:
        """
        Retrieves the top matching crochet patterns based on a semantic query. 
        The number of results returned is controlled by 'num_patterns' with a maximum limit of 10.
        """

        limit = min(num_patterns, 10)
        patterns = self.search_client.search(
            search_text=semantic_query,
            query_type="semantic",
            semantic_configuration_name="default",
            query_caption="extractive",
            query_answer="extractive", 
            top = limit
        )
        result_str = ""
        for pattern in patterns:
            result_str += (
                f"\nPattern Name: {pattern['pattern_name']}\n"
                f"Author: {pattern.get('author', '')}\n"
                f"Category: {pattern.get('category', '')}\n"
                f"Tags: {', '.join(pattern.get('tags', []))}\n\n"
                f"URL: {pattern.get('pattern_url')}\n\n"
            )

        return result_str.strip()

    

