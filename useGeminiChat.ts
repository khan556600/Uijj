import { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Blob, LiveSession } from "@google/genai";
import { encode, decode, decodeAudioData } from '../services/audioUtils';
import { Role, Message, ConnectionState } from '../types';

// Per Gemini API guidelines, create a helper to format audio data.
const createBlob = (data: Float32Array): Blob => {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

export const useGeminiChat = () => {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [currentUserTurn, setCurrentUserTurn] = useState('');
  const [currentModelTurn, setCurrentModelTurn] = useState('');
  const [error, setError] = useState<string | null>(null);

  const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  
  // Refs to accumulate transcription chunks
  const currentInputTranscriptionRef = useRef('');
  const currentOutputTranscriptionRef = useRef('');

  // Ref to manage conversation turn state
  const turnStateRef = useRef<'user' | 'model' | 'idle'>('idle');
  
  // State and effect to queue model output, breaking React's state batching
  // to allow the 'thinking' indicator to render.
  const [queuedModelChunks, setQueuedModelChunks] = useState<string[]>([]);
  useEffect(() => {
    if (queuedModelChunks.length > 0) {
        setCurrentModelTurn(prev => prev + queuedModelChunks.join(''));
        setQueuedModelChunks([]);
    }
  }, [queuedModelChunks]);


  const cleanup = useCallback(() => {
    setError(null);
    sessionPromiseRef.current?.then(session => session.close()).catch(console.error);
    sessionPromiseRef.current = null;
    
    scriptProcessorRef.current?.disconnect();
    scriptProcessorRef.current = null;
    
    mediaStreamSourceRef.current?.disconnect();
    mediaStreamSourceRef.current = null;
    
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
    
    inputAudioContextRef.current?.close().catch(console.error);
    inputAudioContextRef.current = null;

    outputAudioContextRef.current?.close().catch(console.error);
    outputAudioContextRef.current = null;

    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;

    setConnectionState(ConnectionState.DISCONNECTED);
  }, []);


  const startSession = useCallback(async () => {
    setConnectionState(ConnectionState.CONNECTING);
    setError(null);
    setChatHistory([]);
    setCurrentUserTurn('');
    setCurrentModelTurn('');
    turnStateRef.current = 'idle';

    try {
      if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Fenrir' } } },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: 'You are Joseph AI, an AI assistant created by Joseph. When asked your name, you must respond with "My name is Joseph AI, created by Joseph". Maintain a friendly and helpful conversational tone and keep your other responses concise and natural.'
        },
        callbacks: {
          onopen: () => {
            setConnectionState(ConnectionState.CONNECTED);
            if (!inputAudioContextRef.current || !mediaStreamRef.current) return;
            
            mediaStreamSourceRef.current = inputAudioContextRef.current.createMediaStreamSource(mediaStreamRef.current);
            scriptProcessorRef.current = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
            
            scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromiseRef.current?.then(session => {
                session.sendRealtimeInput({ media: pcmBlob });
              }).catch(e => {
                  console.error("Error sending audio input:", e);
                  setError("Failed to send audio. Please restart.");
                  cleanup();
              });
            };
            
            mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
            scriptProcessorRef.current.connect(inputAudioContextRef.current.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.inputTranscription) {
              if (turnStateRef.current === 'idle') {
                turnStateRef.current = 'user';
              }
              currentInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
              setCurrentUserTurn(currentInputTranscriptionRef.current);
            }

            if (message.serverContent?.outputTranscription) {
              if (turnStateRef.current === 'user') {
                  const fullUserInput = currentInputTranscriptionRef.current.trim();
                  if (fullUserInput) {
                      setChatHistory(prev => [...prev, { role: Role.USER, text: fullUserInput }]);
                  }
                  setCurrentUserTurn('');
                  currentInputTranscriptionRef.current = '';
                  turnStateRef.current = 'model';
              }
              currentOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
              setQueuedModelChunks([message.serverContent.outputTranscription.text]);
            }
            
            if (message.serverContent?.turnComplete) {
                const fullModelOutput = currentOutputTranscriptionRef.current.trim();
                const fullUserInput = currentInputTranscriptionRef.current.trim();

                setChatHistory(prev => {
                    const newHistory = [...prev];
                    // Finalize user input if it wasn't already (e.g., user spoke but model didn't)
                    if (turnStateRef.current === 'user' && fullUserInput) {
                      newHistory.push({ role: Role.USER, text: fullUserInput });
                    }
                    // Finalize model output
                    if (fullModelOutput) {
                      newHistory.push({ role: Role.MODEL, text: fullModelOutput });
                    }
                    return newHistory;
                });

                // Reset for the next turn
                currentInputTranscriptionRef.current = '';
                currentOutputTranscriptionRef.current = '';
                setCurrentUserTurn('');
                setCurrentModelTurn('');
                setQueuedModelChunks([]);
                turnStateRef.current = 'idle';
            }

            const interrupted = message.serverContent?.interrupted;
            if (interrupted) {
              for (const source of audioSourcesRef.current) {
                source.stop();
              }
              audioSourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }

            const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData.data;
            if (audioData && outputAudioContextRef.current) {
                const audioContext = outputAudioContextRef.current;
                const audioBuffer = await decodeAudioData(decode(audioData), audioContext, 24000, 1);

                nextStartTimeRef.current = Math.max(nextStartTimeRef.current, audioContext.currentTime);

                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                source.addEventListener('ended', () => {
                    audioSourcesRef.current.delete(source);
                });
                
                source.start(nextStartTimeRef.current);
                nextStartTimeRef.current += audioBuffer.duration;
                audioSourcesRef.current.add(source);
            }
          },
          onerror: (e: ErrorEvent) => {
            console.error(e);
            setError('A connection error occurred. Please try again.');
            cleanup();
          },
          onclose: () => {
            cleanup();
          },
        },
      });

    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to start session.');
      cleanup();
    }
  }, [cleanup]);

  const endSession = useCallback(() => {
    cleanup();
  }, [cleanup]);

  return { connectionState, chatHistory, currentUserTurn, currentModelTurn, startSession, endSession, error };
};