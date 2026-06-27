from loguru import logger

def preload_models():
    """Download and cache AI models before server starts.
    
    This ensures that the first user connection does not stall while heavy 
    models download, providing a seamless startup experience.
    """
    logger.info("[SYSTEM] Initializing AI models. This may take a moment if downloading for the first time...")
    
    try:
        logger.info("[SYSTEM] 1/3 Loading Silero VAD (Voice Activity Detection)...")
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        # Instantiating the class automatically downloads the ONNX model to cache if missing
        _ = SileroVADAnalyzer()
        
        logger.info("[SYSTEM] 2/3 Loading Kokoro TTS (Text-to-Speech)...")
        from pipecat.services.kokoro.tts import KokoroTTSService
        # Instantiating the class automatically downloads Kokoro ONNX and voices file to cache
        _ = KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"))
        
        logger.info("[SYSTEM] 3/3 Loading MLX Whisper (Speech-to-Text)...")
        from huggingface_hub import snapshot_download
        # We explicitly download the MLX Whisper repository to cache
        # because Pipecat's MLX service uses lazy-loading and ignores downloads on init
        snapshot_download(repo_id="mlx-community/distil-whisper-medium.en")
        
        logger.info("[SYSTEM] ✅ All AI models loaded successfully!")
    except Exception as e:
        logger.error(f"[SYSTEM] ❌ Failed to preload models: {e}")
        raise
