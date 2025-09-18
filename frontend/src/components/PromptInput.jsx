import React, { useState, useRef, useEffect } from 'react';
import { FiMic, FiSend } from 'react-icons/fi';

export default function PromptInput({ onSubmit }) {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setIsListening(false);
      setInput('');
      if (transcript) {
        onSubmit(transcript);
      }
    };

    recognitionRef.current.onerror = (event) => {
      console.error('Speech recognition error', event);
      setIsListening(false);
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };
  }, [onSubmit]);

  const handleVoiceClick = () => {
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      setInput('');
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  const handleSubmit = () => {
    if (input.trim()) {
      onSubmit(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="flex items-center w-full px-4 py-2 bg-[#1e1e1e] rounded-full shadow">
      <input
        type="text"
        className="flex-1 bg-transparent outline-none text-white placeholder-gray-400"
        placeholder="Type your prompt here..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button
        onClick={handleVoiceClick}
        className={`ml-2 text-white rounded-full p-2 ${isListening ? 'bg-green-600 animate-pulse' : 'bg-gray-700'}`}
      >
        <FiMic size={18} />
      </button>
      <button
        onClick={handleSubmit}
        className="ml-2 bg-green-500 hover:bg-green-600 text-white rounded-full p-2"
      >
        <FiSend size={18} />
      </button>
    </div>
  );
}
