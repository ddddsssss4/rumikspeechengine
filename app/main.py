import os
import time

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Optional

from contextlib import asynccontextmanager

from app.preload import preload_models
from app.voice_agent import run_voice_agent

load_dotenv(override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload models before accepting traffic
    preload_models()
    yield
    # Cleanup on shutdown (if needed)

app = FastAPI(title="NeuralEngine Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenRequest(BaseModel):
    room_name: Optional[str] = None
    participant_identity: Optional[str] = None
    participant_name: Optional[str] = None
    participant_metadata: Optional[str] = None
    participant_attributes: Optional[Dict[str, str]] = None
    room_config: Optional[dict] = None
    tts_service: Optional[str] = "kokoro"


def _generate_bot_token(room_name: str) -> str:
    """Mint a LiveKit JWT for the voice agent bot."""
    token = (
        api.AccessToken(
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        .with_identity("voice-agent-bot")
        .with_name("AI Assistant")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    return token.to_jwt()


@app.get("/")
def health():
    return {"status": "ok", "service": "neuralengine-voice-agent"}


@app.post("/api/token", status_code=201)
async def get_token(request: TokenRequest, background_tasks: BackgroundTasks):
    try:
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        server_url = os.getenv("LIVEKIT_URL")

        if not all([api_key, api_secret, server_url]):
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: missing LiveKit credentials",
            )

        room_name = request.room_name or f"room-{int(time.time())}"
        participant_identity = request.participant_identity or f"user-{int(time.time())}"
        participant_name = request.participant_name or "User"

        # Mint token for the human participant
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(participant_identity)
            .with_name(participant_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
        )

        if request.participant_metadata:
            token = token.with_metadata(request.participant_metadata)
        if request.participant_attributes:
            token = token.with_attributes(request.participant_attributes)
        if request.room_config:
            token = token.with_room_config(request.room_config)

        participant_token = token.to_jwt()

        # Mint a separate token for the bot and start the voice agent
        bot_token = _generate_bot_token(room_name)
        background_tasks.add_task(
            run_voice_agent,
            url=server_url,
            token=bot_token,
            room_name=room_name,
            tts_service_type=request.tts_service,
        )
        logger.info(f"Voice agent spawned for room: {room_name}")

        return {
            "server_url": server_url,
            "participant_token": participant_token,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate token")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
