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
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
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
RUMIK_SYSTEM_PROMPT = """You are a helpful voice assistant. Your responses will be spoken aloud using the Silk Muga 1 text-to-speech model.

Rules:
- Output only the final tagged text, no markdown or notes.
- Romanised Hinglish only (Latin script). Never Devanagari.
- Start every paragraph with one tone tag, as the first token:
  [happy], [excited], [sad], [angry], [neutral], [whisper].
- Keep replies short: 1 to 2 sentences.
- Respond to what the user said in a creative, helpful, and brief way.
- Avoid emojis, bullet points, or other formatting that can't be spoken."""

# Standard prompt for Kokoro English TTS
KOKORO_SYSTEM_PROMPT = """You are a helpful voice assistant. Your responses will be spoken aloud.

Rules:
- Output only the final spoken text, no markdown, asterisks, or notes.
- Keep replies short: 1 to 2 sentences.
- Respond to what the user said in a creative, helpful, and brief way.
- Avoid emojis, bullet points, or other formatting that can't be spoken."""



class TranscriptionObserver(BaseObserver):
    """Observes pipeline frames and forwards transcription data to the LiveKit room.

    Hooks into the pipecat observer system (the correct API for non-intrusive
    frame inspection in pipecat 1.4.0) to:
      - Forward STT TranscriptionFrame (user speech) via send_text
      - Stream LLM token output via stream_text (open → write chunks → close)
    """

    def __init__(self, transport: LiveKitTransport):
        super().__init__()
        self._transport = transport
        # Mutable dict so the writer reference survives across on_push_frame calls.
        self._llm_state: dict = {"writer": None}

    @property
    def _room(self):
        return self._transport._client.room

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        room = self._room
        if not room:
            return

        # ── User speech ──────────────────────────────────────────────────────
        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"[Observer] TranscriptionFrame: {frame.text!r}")
            await room.local_participant.send_text(
                frame.text,
                topic="lk.transcription",
                attributes={
                    "lk.transcribed_track_id": frame.user_id,
                    "lk.transcription_final": "true",
                    "speaker": "user",
                },
            )

        # ── Agent LLM streaming ──────────────────────────────────────────────
        elif isinstance(frame, LLMFullResponseStartFrame):
            # Open one persistent TextStreamWriter; all token chunks share the
            # same stream ID so the client updates a single bubble incrementally.
            writer = room.local_participant.stream_text(
                topic="lk.transcription",
                attributes={
                    "lk.transcription_final": "false",
                    "speaker": "agent",
                },
            )
            self._llm_state["writer"] = writer
            logger.debug("[Observer] Opened LLM text stream")

        elif isinstance(frame, LLMTextFrame):
            writer = self._llm_state.get("writer")
            if writer is not None:
                await writer.write(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            writer = self._llm_state.get("writer")
            if writer is not None:
                await writer.aclose()
                self._llm_state["writer"] = None
                logger.debug("[Observer] Closed LLM text stream")


async def run_voice_agent(url: str, token: str, room_name: str, tts_service_type: str = "kokoro"):
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
        )
    )

    # Choose TTS and Prompt
    tts_service_type = tts_service_type.lower()
    if tts_service_type == "rumik":
        system_prompt = RUMIK_SYSTEM_PROMPT
        tts = RumikTTSService(
            api_key=os.environ["RUMIK_API_KEY"],
            gateway_url=os.environ["RUMIK_GATEWAY_URL"],
            settings=RumikTTSService.Settings(model="muga"),
        )
        logger.info("[BOT] Using Rumik TTS")
    else:
        system_prompt = KOKORO_SYSTEM_PROMPT
        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice="af_heart",
            )
        )
        logger.info("[BOT] Using Kokoro TTS")

    # --- LLM: Local via LM Studio (OpenAI-compatible API) ---
    llm = OpenAILLMService(
        api_key="local",  # LM Studio doesn't require a real key
        base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        settings=OpenAILLMService.Settings(
            model=os.environ.get("LM_STUDIO_MODEL", "qwen1.5-0.5b-chat"),
            system_instruction=system_prompt,
        ),
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
                    stop_secs=0.2,    # Revert to 0.2s so TurnAnalyzer doesn't time out
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

    # Wire up the observer — this is the correct pipecat 1.4.0 API for
    # intercepting frames without injecting processors into the pipeline.
    transcription_observer = TranscriptionObserver(transport)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            observers=[transcription_observer],
        ),
    )


    # Greet the user when they join and warm up Whisper
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        logger.info(f"[BOT] First participant joined: {participant_id}")
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
