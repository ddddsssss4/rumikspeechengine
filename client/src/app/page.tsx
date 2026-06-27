'use client';

import { useEffect, useRef, useState } from 'react';
import {
  Room,
  RoomEvent,
  createLocalAudioTrack,
  RemoteTrack,
  Track,
  TranscriptionSegment,
  Participant
} from 'livekit-client';

interface TranscriptMsg {
  id: string;
  text: string;
  isAgent: boolean;
  isFinal: boolean;
}

export default function Home() {
  const [room, setRoom] = useState<Room | null>(null);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptMsg[]>([]);
  const [ttsService, setTtsService] = useState<'kokoro' | 'rumik'>('kokoro');
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of transcripts
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcripts]);

  const joinRoom = async () => {
    try {
      setStatus('connecting');
      setError(null);
      setTranscripts([]);
      
      const res = await fetch('http://localhost:8000/api/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          room_name: 'neural-voice', 
          participant_name: 'user',
          tts_service: ttsService 
        }),
      });

      if (!res.ok) {
        throw new Error('Failed to fetch token from backend');
      }

      const data = await res.json();
      const newRoom = new Room();

      newRoom.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind === Track.Kind.Audio && audioRef.current) {
          track.attach(audioRef.current);
        }
      });

      newRoom.registerTextStreamHandler('lk.transcription', async (reader, participant) => {
        try {
          const segmentId = reader.info.id || Math.random().toString(36).substring(7);
          // Use the explicit speaker attribute — both user & agent streams are sent
          // by the bot participant, so participant.identity alone cannot distinguish them.
          const isAgent = reader.info.attributes?.['speaker'] === 'agent';
          const isFinalStream = reader.info.attributes?.['lk.transcription_final'] !== 'false';

          // Add an empty placeholder immediately so the bubble appears as soon
          // as the stream opens (no pop-in after the full text arrives).
          setTranscripts(prev => {
            if (prev.some(t => t.id === segmentId)) return prev;
            return [...prev, { id: segmentId, text: '', isAgent, isFinal: false }];
          });

          // Iterate chunks as they arrive from the LiveKit TextStreamReader
          // so the bubble grows token-by-token in real-time.
          for await (const chunk of reader) {
            setTranscripts(prev =>
              prev.map(t =>
                t.id === segmentId ? { ...t, text: t.text + chunk } : t
              )
            );
          }

          // Stream is fully closed — mark the bubble as final.
          setTranscripts(prev =>
            prev.map(t =>
              t.id === segmentId ? { ...t, isFinal: isFinalStream } : t
            )
          );
        } catch (e) {
          console.error("Error reading transcript:", e);
        }
      });

      newRoom.on(RoomEvent.Disconnected, () => {
        setRoom(null);
        setStatus('idle');
      });

      await newRoom.connect(data.server_url, data.participant_token);
      const micTrack = await createLocalAudioTrack();
      await newRoom.localParticipant.publishTrack(micTrack);

      setRoom(newRoom);
      setStatus('connected');
    } catch (err: any) {
      console.error('Connection error:', err);
      setError(err.message || 'An error occurred while connecting');
      setStatus('idle');
    }
  };

  const leaveRoom = async () => {
    await room?.disconnect();
    setRoom(null);
    setStatus('idle');
  };

  return (
    <main className="min-h-screen p-8 md:p-12 lg:p-24 flex justify-center selection:bg-black selection:text-[#f4f1eb]">
      <div className="w-full max-w-3xl flex flex-col">
        
        {/* Main outer border container matching the reference style */}
        <div className="border-4 border-black rounded-sm overflow-hidden bg-transparent">
          
          {/* Header block */}
          <div className="border-b-4 border-black p-6 md:p-8 bg-transparent">
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-black">
              100% Local Voice Agent Pipeline
            </h1>
          </div>

          {/* Section 1 */}
          <div className="border-b-4 border-black p-6 md:p-8 bg-transparent">
            <div className="flex items-baseline mb-6">
              <h2 className="text-3xl font-black text-black tracking-tight mr-4">1. STT & LLM Engine</h2>
              <span className="text-black font-medium">/ whisper + lm-studio /</span>
            </div>
            
            <p className="text-black text-lg md:text-xl font-medium leading-relaxed max-w-2xl mb-12">
              Powered by Pipecat and LiveKit WebRTC. Audio is streamed in real-time, transcribed locally via Whisper MLX, and processed by a local LLM in LM Studio.
            </p>

            {/* The Big Rounded Rectangle with Waveform */}
            <div className="border-[3px] border-black rounded-3xl p-8 flex flex-col items-center justify-center relative min-h-[300px] mb-8 bg-transparent transition-colors duration-500">
              
              {/* Waveform Visualization */}
              <div className="flex items-center justify-center h-40 w-full mb-8">
                {status === 'connected' ? (
                  <div className="flex items-center space-x-2 h-full">
                    {/* Animated bars when connected */}
                    <div className="w-6 bg-black rounded-full bar" style={{animationDuration: '0.8s'}} />
                    <div className="w-8 bg-black rounded-full bar" style={{animationDuration: '1.2s'}} />
                    <div className="w-10 bg-black rounded-full bar" style={{animationDuration: '0.9s'}} />
                    <div className="w-14 bg-black rounded-full bar" style={{animationDuration: '1.5s'}} />
                    <div className="w-10 bg-black rounded-full bar" style={{animationDuration: '1.1s'}} />
                    <div className="w-8 bg-black rounded-full bar" style={{animationDuration: '1.3s'}} />
                    <div className="w-6 bg-black rounded-full bar" style={{animationDuration: '0.7s'}} />
                  </div>
                ) : (
                  <div className="flex items-center space-x-2 h-full opacity-90">
                    {/* Static waveform matching the image */}
                    <div className="w-8 h-8 bg-black rounded-full" />
                    <div className="w-6 h-4 bg-black rounded-full" />
                    <div className="w-12 h-16 bg-black rounded-full" />
                    <div className="w-10 h-28 bg-black rounded-full" />
                    <div className="w-12 h-32 bg-black rounded-full" />
                    <div className="w-10 h-24 bg-black rounded-full" />
                    <div className="w-12 h-16 bg-black rounded-full" />
                    <div className="w-10 h-10 bg-black rounded-full" />
                  </div>
                )}
              </div>

              {/* TTS Service Selector */}
              <div className="flex items-center justify-center space-x-4 mb-8">
                <button 
                  onClick={() => setTtsService('kokoro')}
                  disabled={status !== 'idle'}
                  className={`px-6 py-2 border-2 border-black font-bold uppercase tracking-widest text-sm rounded-l-full transition-colors ${ttsService === 'kokoro' ? 'bg-black text-[#f4f1eb]' : 'bg-transparent text-black hover:bg-black/10'} ${status !== 'idle' ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  Kokoro (Local)
                </button>
                <button 
                  onClick={() => setTtsService('rumik')}
                  disabled={status !== 'idle'}
                  className={`px-6 py-2 border-2 border-black font-bold uppercase tracking-widest text-sm rounded-r-full transition-colors ${ttsService === 'rumik' ? 'bg-black text-[#f4f1eb]' : 'bg-transparent text-black hover:bg-black/10'} ${status !== 'idle' ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  Rumik (Muga)
                </button>
              </div>

              {/* Status / Call Button */}
              {status === 'connecting' ? (
                <div className="px-10 py-3 rounded-full border-2 border-black text-black font-semibold text-lg flex items-center bg-[#f4f1eb]">
                  <span className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin mr-3"></span>
                  connecting...
                </div>
              ) : (
                <button
                  onClick={status === 'connected' ? leaveRoom : joinRoom}
                  className="px-12 py-2.5 rounded-full border-[2.5px] border-black text-black font-bold text-lg hover:bg-black hover:text-[#f4f1eb] transition-colors bg-transparent"
                >
                  {status === 'connected' ? 'end call' : 'call now'}
                </button>
              )}
              
              {error && (
                <p className="mt-4 text-red-600 font-bold text-sm bg-red-100 px-4 py-2 rounded-md border-2 border-red-900 absolute bottom-4">
                  {error}
                </p>
              )}
            </div>

            {/* Transcription Box */}
            <div className="mt-4">
              <h3 className="font-bold text-black mb-4 uppercase tracking-widest text-sm flex items-center">
                <div className="w-2 h-2 bg-black rounded-full mr-2" />
                Live Transcription
              </h3>
              <div 
                ref={scrollRef}
                className="h-64 border-[3px] border-black rounded-xl p-6 overflow-y-auto custom-scrollbar flex flex-col gap-5 bg-transparent"
              >
                {transcripts.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-black/40 font-medium italic">
                    {status === 'connected' ? 'Listening...' : 'Transcriptions will appear here.'}
                  </div>
                ) : (
                  transcripts.map((t) => (
                    <div key={t.id} className={`flex flex-col ${t.isAgent ? 'items-start' : 'items-end'}`}>
                      <div className="text-[11px] font-extrabold mb-1 uppercase tracking-widest text-black/60">
                        {t.isAgent ? `Agent (${ttsService === 'kokoro' ? 'Kokoro' : 'Rumik'})` : 'You'}
                      </div>
                      <div className={`
                        px-5 py-3 text-lg font-medium leading-relaxed max-w-[85%] text-black border-2 border-black
                        ${t.isAgent ? 'rounded-2xl rounded-tl-sm bg-white' : 'rounded-2xl rounded-tr-sm bg-black text-white'}
                        ${!t.isFinal ? 'opacity-70 animate-pulse' : 'opacity-100'}
                      `}>
                        {t.text}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>

          {/* Section 2 */}
          <div className="p-6 md:p-8 bg-transparent">
            <div className="flex items-baseline mb-4">
              <h2 className="text-3xl font-black text-black tracking-tight mr-4">2. TTS & VAD</h2>
              <span className="text-black font-medium">/ kokoro + silero /</span>
            </div>
            <p className="text-black text-lg md:text-xl font-medium leading-relaxed max-w-2xl">
              Utterances are intelligently chunked and synthesized on-device using Kokoro TTS, triggered seamlessly by Silero's VAD models.
            </p>
          </div>

        </div>
      </div>
      
      <audio ref={audioRef} autoPlay />
    </main>
  );
}
