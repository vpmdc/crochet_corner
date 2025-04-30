import os
import asyncio
import chainlit as cl
import json
from datetime import datetime, timezone
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.filters import FunctionInvocationContext
from plugins.youtube_video_pattern import YoutubePatternExtracterPlugin
from plugins.crochet_tutor import CrochetTutorPlugin
from plugins.pattern_search import PatternSearchPlugin
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.filters import FunctionInvocationContext


kernel = Kernel()

async def function_invocation_filter(context: FunctionInvocationContext, next):
    """Filter that logs function invocations and their results"""

    # Extract the function name, "user_message" or fallback to default
    function_name = context.function.name if context.function else "Unknown"
    args = context.arguments.get("user_message", "[no 'user_message' argument]")

    # Format arguments for display
    try:
        args_display = json.dumps(args, indent=4, default=str) if isinstance(args, (dict, list)) else str(args)
    except Exception:
        args_display = str(args)


    current_step = cl.Step(name=function_name, type="tool")
    current_step.language = "json"
    current_step.start = datetime.now(timezone.utc).isoformat()

    # Log the function name and input arguments
    await current_step.stream_token(f"Function Name: `{function_name}`\n")
    await current_step.stream_token("Arguments:")
    await current_step.stream_token(f"{args_display}\n")

    # Execute function
    await next(context)

    # Log result
    try:
        result_value = context.result.value
        result_display = json.dumps(result_value, indent=4, default=str)
    except Exception:
        result_value = context.result.value if context.result else "[no result]"
        result_display = str(result_value)

    await current_step.stream_token("Result:\n")
    await current_step.stream_token(f"```\n{result_display}\n```")

    # End timestamp
    current_step.end = datetime.now(timezone.utc).isoformat()
    await current_step.send()


kernel.add_filter("function_invocation", function_invocation_filter)
azure_openai_lock = asyncio.Lock()
thread: ChatHistoryAgentThread = None
triage_agent = ChatCompletionAgent(
    service=AzureChatCompletion(
        deployment_name="gpt-4o-mini",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ),
    kernel=kernel,
    name="TriageAgent",
    instructions=(
        """Your role is to evaluate the user's request and forward it to the appropriate agent. You have the following plugins:
            - PatternSearchPlugin: Use this when the user wants to find existing crochet patterns based on specific criteria, such as project type, difficulty, or style.
            - YoutubePatternExtracterPlugin: Use this when the user provides a YouTube video link and wants to extract a crochet pattern or instructions from the video.
            - CrochetTutorPlugin: Use this when the user asks questions about crochet stitches, techniques, terminology, or general crochet advice.
            
            Carefully analyze each user request. Select the plugin that best matches the user's needs, and forward the request to that plugin for processing. 
            If a request could fit more than one plugin, choose the one that will provide the most direct and helpful answer.
            If the user's question is not related to crochet, politely inform them that you are only able to assist with crochet related topics and cannot answer unrelated questions."""
    ),
    plugins=[YoutubePatternExtracterPlugin(), CrochetTutorPlugin(), PatternSearchPlugin()],
)

@cl.on_message
async def on_message(message: cl.Message):
    try:
        async with azure_openai_lock:
            response = await triage_agent.get_response(
                messages=message.content,
                thread=thread,
            )
        await cl.Message(content=str(response)).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ An error occurred: {e}").send()



@cl.set_starters
async def set_starters():
    """Sets predefined starter prompts demonstrating main functionalities"""
    return [
        cl.Starter(
            label="Pattern Finder",
            message="Help me find a pattern",
            icon="/public/icons/search_icon.svg",
            ),

        cl.Starter(
            label="Extract Youtube Pattern",
            message="Transcribe the pattern for me from a youtube video",
            icon="/public/icons/transcribe_icon.svg",
            ),
        cl.Starter(
            label="Crochet Help",
            message="I need some crochet help",
            icon="/public/icons/question_mark_icon.svg",
            ),
        ]

