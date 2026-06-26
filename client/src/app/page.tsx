'use client';

import { useEffect, useRef, useState } from 'react';
import {
  Room,
  RoomEvent,
  createLocalAudioTrack,
  RemoteTrack,
  Track,
} from 'livekit-client';
import { Mic, MicOff, Loader2, Sparkles, Activity } from 'lucide-react';

export default function Home() {
  const [room, setRoom] = useState<Room | null>(null);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected'>('idle');
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const joinRoom = async () => {
    try {
      setStatus('connecting');
      setError(null);
      
      const res = await fetch('http://localhost:8000/api/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room_name: 'neural-voice', participant_name: 'user' }),
      });

      if (!res.ok) {
        throw new Error('Failed to fetch token from backend');
      }

      const data = await res.json();
      const newRoom = new Room();

      // Debug: confirm bot's audio track arrives
      newRoom.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        console.log('[LiveKit] TrackSubscribed:', track.kind, track.sid);
        if (track.kind === Track.Kind.Audio && audioRef.current) {
          track.attach(audioRef.current);
        }
      });

      // Debug: confirm our mic was published
      newRoom.on(RoomEvent.LocalTrackPublished, (pub) => {
        console.log('[LiveKit] LocalTrackPublished:', pub.kind, pub.trackSid);
      });

      newRoom.on(RoomEvent.Disconnected, () => {
        setRoom(null);
        setStatus('idle');
      });

      await newRoom.connect(data.server_url, data.participant_token);

      // Publish microphone
      const micTrack = await createLocalAudioTrack();
      await newRoom.localParticipant.publishTrack(micTrack);
      console.log('[LiveKit] Mic published');

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
    <main className="flex-1 flex flex-col items-center justify-center p-6 relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden z-[-1]">
        <div className="absolute top-[20%] left-[10%] w-[40rem] h-[40rem] bg-orange-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute bottom-[20%] right-[10%] w-[30rem] h-[30rem] bg-blue-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '10s' }} />
      </div>

      <div className="max-w-md w-full glass-panel rounded-3xl p-8 flex flex-col items-center justify-center space-y-10 animate-float shadow-2xl border border-white/10 relative z-10">
        
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center p-2 bg-orange-500/20 text-orange-400 rounded-full mb-4 ring-1 ring-orange-500/30 shadow-[0_0_15px_rgba(249,115,22,0.2)]">
            <Sparkles className="w-4 h-4 mr-2" />
            <span className="text-xs font-semibold uppercase tracking-wider">NeuralEngine</span>
          </div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-100 to-gray-500">
            Voice Agent
          </h1>
          <p className="text-sm text-gray-400">Powered by Pipecat & Rumik AI</p>
        </div>

        {/* The Interaction Orb */}
        <div className="relative flex items-center justify-center h-48 w-48">
          {status === 'connected' && (
            <div className="absolute inset-0 ripple rounded-full pointer-events-none" />
          )}
          
          <button
            onClick={status === 'connected' ? leaveRoom : joinRoom}
            disabled={status === 'connecting'}
            className={`relative flex items-center justify-center w-32 h-32 rounded-full transition-all duration-500 shadow-2xl focus:outline-none focus:ring-4 focus:ring-primary/30 group ${
              status === 'connected' ? 'orb scale-105' : 'bg-gray-800/80 hover:bg-gray-700/80 border border-white/5'
            }`}
          >
            {status === 'connecting' ? (
              <Loader2 className="w-10 h-10 text-white animate-spin" />
            ) : status === 'connected' ? (
              <div className="flex flex-col items-center">
                <Mic className="w-10 h-10 text-white mb-2" />
                <div className="flex items-end justify-center space-x-1 h-4">
                  <div className="w-1 bg-white/80 rounded-full bar" />
                  <div className="w-1 bg-white/80 rounded-full bar" />
                  <div className="w-1 bg-white/80 rounded-full bar" />
                  <div className="w-1 bg-white/80 rounded-full bar" />
                  <div className="w-1 bg-white/80 rounded-full bar" />
                </div>
              </div>
            ) : (
              <MicOff className="w-10 h-10 text-gray-400 group-hover:text-white transition-colors duration-300" />
            )}
          </button>
        </div>

        {/* Status Text & Errors */}
        <div className="h-12 flex flex-col items-center justify-center text-center">
          {error ? (
            <p className="text-red-400 text-sm bg-red-400/10 py-1.5 px-3 rounded-full border border-red-400/20">
              {error}
            </p>
          ) : (
            <div className="flex items-center text-sm font-medium">
              {status === 'idle' && <span className="text-gray-400">Tap the mic to connect</span>}
              {status === 'connecting' && <span className="text-orange-400 animate-pulse">Initializing pipeline...</span>}
              {status === 'connected' && (
                <div className="flex items-center text-green-400">
                  <div className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-ping" />
                  Connected & Listening
                </div>
              )}
            </div>
          )}
        </div>

      </div>

      {/* Audio element for Pipecat TTS playback */}
      <audio ref={audioRef} autoPlay />
    </main>
  );
}
