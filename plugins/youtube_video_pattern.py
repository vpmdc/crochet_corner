import os
from typing import Annotated
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from openai import AzureOpenAI
from semantic_kernel.functions import kernel_function


INFO_EXTRACTOR_SYSTEM_MESSAGE = """
You are a helpful AI assistant that specializes in extracting crochet patterns from YouTube tutorial videos.

Task:

Given a transcript from a YouTube video.
If it is a crochet tutorial, extract and summarize the full crochet pattern in a clear, step-by-step format.
If it is not a valid transcript for a youtube video, return message claiming no pattern can be found.
List all required tools and materials separately, including yarn types, hook sizes, and any accessories mentioned.
If specific yarn brands, colors, or amounts are mentioned, include them.
If any details are missing or unclear, reasonably infer based on standard crochet practices but clearly label inferred items as "(inferred)".

Organize your output into two sections:

- Materials & Tools List
- Full Crochet Pattern Description (Step-by-Step)

Important:

Only use information verifiable from the video, transcript, or description unless inferring is necessary.
Keep the tone instructional, beginner-friendly, and neatly formatted.
Always make sure your output is clean, consistent, and easy to follow.
"""

INFO_EXTRACTOR_USER_MESSAGE = """
Input: "{transcript}"
Output:
"""


class YoutubePatternExtracterPlugin:
    """Plugin to extract crochet patterns from YouTube video transcripts."""

    @kernel_function
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.INFO_EXTRACTOR_SYSTEM_MESSAGE = INFO_EXTRACTOR_SYSTEM_MESSAGE
        self.INFO_EXTRACTOR_USER_MESSAGE = INFO_EXTRACTOR_USER_MESSAGE

    @kernel_function
    def extract_video_id(self, url: str) -> str:
        parsed_url = urlparse(url)
        if 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
        if 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        return None

    @kernel_function
    def get_youtube_transcript(self, video_id: str) -> str:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript])

    @kernel_function(description="Extracts a crochet pattern from a YouTube video URL")
    def extract_pattern_from_youtube(
        self,
        video_url: Annotated[str, "A full YouTube video URL"]
    ) -> Annotated[str, "The extracted crochet pattern, if found"]:
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return "Invalid YouTube URL"

        try:
            transcript = self.get_youtube_transcript(video_id)
        except Exception as e:
            return f"Failed to retrieve transcript: {str(e)}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with your deployed model
                messages=[
                    {"role": "system", "content": self.INFO_EXTRACTOR_SYSTEM_MESSAGE},
                    {"role": "user", "content": self.INFO_EXTRACTOR_USER_MESSAGE.replace('{transcript}', transcript)}
                ],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to process transcript with AI: {str(e)}"

