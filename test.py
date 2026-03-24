#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.google.tts import GeminiTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# Flow nodes
def create_initial_node() -> NodeConfig:
    """Create the initial node of the flow.

    Define the bot's role and task for the node as well as the function for it to call.
    The function call includes a handler which provides the function call result to
    Pipecat and then transitions to the next node.
    """
    record_favorite_color_func = FlowsFunctionSchema(
        name="record_favorite_color_func",
        description="Record the color the user said is their favorite.",
        required=["color"],
        handler=record_favorite_color_and_set_next_node,
        properties={"color": {"type": "string"}},
    )

    return {
        "name": "initial",
        "role_message": "Always speak in English and in 3-4 sentences. Your responses will be converted to audio. Avoid outputting special characters and emojis.",
        "task_messages": [
            {
                "role": "system",
                "content": "Greet the user in English in 3-4 sentences. Start the conversation in a friendly way.",
            }
        ],
        "functions": [record_favorite_color_func],
    }


async def record_favorite_color_and_set_next_node(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Function handler that records the color then sets the next node.

    Here "record" means print to the console, but any logic could go here;
    Write to a database, make an API call, etc.
    """
    print(f"Your favorite color is: {args['color']}")
    return args["color"], create_end_node()


def create_end_node() -> NodeConfig:
    """End the conversation.

    Flows transitions to this node when the user has answered the question.
    It thanks the user and ends the conversation using the `end_conversation`
    post-action.
    """
    return NodeConfig(
        name="create_end_node",
        task_messages=[
            {
                "role": "user",
                "content": "Thank the user for answering and end the conversation",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        live_options=LiveOptions(model=os.getenv("DEEPGRAM_MODEL", "nova-2")),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-5.3-chat-latest",
    )
    cred_json = os.getenv("VERTEX_JSON")
    tts = GeminiTTSService(
        credentials=cred_json,
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
        model="gemini-2.5-pro-tts",
        voice_id="Kore",
        params=GeminiTTSService.InputParams(
            prompt="Read this in a friendly and helpful tone:",
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Initialize flow manager
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
