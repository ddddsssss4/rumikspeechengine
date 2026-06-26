import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.workers.runner import WorkerRunner
from pipecat_rumik import RumikTTSService
from pipecat.services.kokoro.tts import KokoroTTSService

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Muga tone-tag system prompt — tells the LLM to format output
# so Rumik's Muga voice model can apply the right emotion.
# SYSTEM_PROMPT = """You are a helpful voice assistant. Your responses will be spoken aloud using the Silk Muga 1 text-to-speech model.
# 
# Rules:
# - Output only the final tagged text, no markdown or notes.
# - Romanised Hinglish only (Latin script). Never Devanagari.
# - Start every paragraph with one tone tag, as the first token:
#   [happy], [excited], [sad], [angry], [neutral], [whisper].
# - Keep replies short: 1 to 2 sentences.
# - Respond to what the user said in a creative, helpful, and brief way.
# - Avoid emojis, bullet points, or other formatting that can't be spoken."""

# Standard prompt for Kokoro English TTS
SYSTEM_PROMPT = """You are a helpful voice assistant. Your responses will be spoken aloud.

Rules:
- Output only the final spoken text, no markdown, asterisks, or notes.
- Keep replies short: 1 to 2 sentences.
- Respond to what the user said in a creative, helpful, and brief way.
- Avoid emojis, bullet points, or other formatting that can't be spoken."""


async def _warmup_whisper(stt: WhisperSTTServiceMLX):
    """Pre-load the Whisper MLX model so first-utterance latency is low.

    Runs a single transcription on a tiny block of silence. This forces
    the model weights to be loaded and compiled on the Apple Silicon Neural
    Engine before the user actually speaks.
    """
    try:
        import numpy as np
        # 0.5 s of silence at 16 kHz
        silence = (np.zeros(8000, dtype=np.float32) * 32768).astype(np.int16).tobytes()
        await stt._client.transcribe(silence, language="en")
        logger.info("[BOT] Whisper MLX warmup complete ✅")
    except Exception as e:
        logger.warning(f"[BOT] Whisper warmup skipped (non-fatal): {e}")


async def run_voice_agent(url: str, token: str, room_name: str):
    """Start and run the voice agent pipeline in a LiveKit room.

    Args:
        url: LiveKit server WebSocket URL.
        token: JWT token for the bot participant.
        room_name: LiveKit room to join.
    """
    logger.info(f"Starting voice agent in room: {room_name}")

    # --- Transport: LiveKit (audio-only) ---
    transport = LiveKitTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # auto_subscribe is hardcoded True inside LiveKitTransportClient.connect()
        ),
    )

    # --- STT: Local Whisper on Apple Silicon ---
    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model="mlx-community/whisper-small-mlx-q4",  # MLX-native quantised model
        ),
    )

    # --- LLM: Local via LM Studio (OpenAI-compatible API) ---
    llm = OpenAILLMService(
        api_key="local",  # LM Studio doesn't require a real key
        base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        settings=OpenAILLMService.Settings(
            model=os.environ.get("LM_STUDIO_MODEL", "google/gemma-4-e4b"),
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    # --- TTS: Rumik AI (Muga voice) ---
    # rumik_tts = RumikTTSService(
    #     api_key=os.environ["RUMIK_API_KEY"],
    #     gateway_url=os.environ["RUMIK_GATEWAY_URL"],
    #     settings=RumikTTSService.Settings(model="muga"),
    # )

    # --- TTS: Kokoro Local TTS ---
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice="af_heart",
        )
    )

    # --- Context management ---
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.4,   # Default 0.7 is too strict for browser WebRTC
                    min_volume=0.3,   # Default 0.6 is way too high for browser mic
                    start_secs=0.2,
                    stop_secs=0.8,    # Give more time after speech ends
                )
            )
        ),
    )

    # --- Pipeline: mic → STT → LLM → TTS → speaker ---
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Greet the user when they join and warm up Whisper
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        logger.info(f"[BOT] First participant joined: {participant_id}")
        # Warm up Whisper MLX by running a silent transcription
        # so the model is already loaded before the user speaks
        logger.info("[BOT] Warming up Whisper MLX model...")
        asyncio.create_task(_warmup_whisper(stt))
        await asyncio.sleep(1)
        await worker.queue_frame(
            TTSSpeakFrame("Hello there! I am your AI assistant. How can I help you today?")
        )

    @transport.event_handler("on_audio_track_subscribed")
    async def on_audio_track_subscribed(transport, participant_id):
        logger.info(f"[BOT] ✅ Audio track subscribed from participant: {participant_id}")

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant_id):
        logger.info(f"[BOT] Participant joined room: {participant_id}")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant_id, reason):
        logger.info(f"[BOT] Participant left room: {participant_id} (reason: {reason})")

    # Handle text chat messages from the LiveKit room
    @transport.event_handler("on_data_received")
    async def on_data_received(transport, data, participant_id):
        logger.info(f"Received data from participant {participant_id}: {data}")
        json_data = json.loads(data)

        await worker.queue_frames(
            [
                InterruptionFrame(),
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    user_id=participant_id,
                    timestamp=json_data["timestamp"],
                    text=json_data["message"],
                ),
                UserStoppedSpeakingFrame(),
            ],
        )

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()
