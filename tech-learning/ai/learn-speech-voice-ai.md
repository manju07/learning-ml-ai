# Speech & Voice AI: Complete Guide

## Table of Contents
1. [Introduction to Speech AI](#introduction-to-speech-ai)
2. [Automatic Speech Recognition (ASR)](#automatic-speech-recognition-asr)
3. [Text-to-Speech (TTS)](#text-to-speech-tts)
4. [Speaker Diarization](#speaker-diarization)
5. [Voice Cloning and Custom Voices](#voice-cloning-and-custom-voices)
6. [Voice Assistants and Conversational AI](#voice-assistants-and-conversational-ai)
7. [Speech Emotion Recognition](#speech-emotion-recognition)
8. [Practical Examples](#practical-examples)
9. [Best Practices](#best-practices)

---

## Introduction to Speech AI

**Speech AI** covers technologies that process and generate human speech: recognition (ASR), synthesis (TTS), understanding, and conversational interfaces.

### Key Applications

| Application | Technology | Use Case |
|--------------|------------|----------|
| **Transcription** | ASR | Meetings, podcasts, accessibility |
| **Voice assistants** | ASR + NLU + TTS | Alexa, Siri, custom agents |
| **Voice cloning** | TTS + cloning | Audiobooks, dubbing |
| **Call center AI** | ASR + NLU | Sentiment, summarization |
| **Real-time captioning** | Streaming ASR | Live events |
| **Voice biometrics** | Speaker recognition | Authentication |

### Pipeline Overview

```
Speech → ASR → Text → NLU/LLM → Response Text → TTS → Speech
```

---

## Automatic Speech Recognition (ASR)

### Whisper (OpenAI)

State-of-the-art ASR; multilingual; robust to accents and noise.

```python
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large
result = model.transcribe("audio.mp3")
print(result["text"])

# With options
result = model.transcribe(
    "audio.mp3",
    language="en",
    task="transcribe",  # or "translate" to English
    fp16=False
)
```

### Hugging Face Transformers

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
result = pipe("audio.wav")
print(result["text"])
```

### Streaming ASR

For real-time: process audio chunks, use streaming models (e.g., faster-whisper, Whisper with chunking).

```python
# faster-whisper: CTranslate2 backend, faster inference
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
```

### Timestamps and Word-Level Alignments

```python
result = model.transcribe("audio.mp3", word_timestamps=True)
for segment in result["segments"]:
    print(f"{segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}")
    if "words" in segment:
        for w in segment["words"]:
            print(f"  {w['word']}: {w['start']}-{w['end']}")
```

### Custom Vocabulary / Forced Alignment

For domain terms (names, products):
- Use custom vocabulary
- Post-process with forced aligner (e.g., Montreal Forced Aligner)
- Or fine-tune Whisper on domain data

---

## Text-to-Speech (TTS)

### OpenAI TTS

```python
from openai import OpenAI
client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    input="Hello, this is a test of text-to-speech."
)
response.stream_to_file("output.mp3")
```

### ElevenLabs

```python
from elevenlabs import ElevenLabs
client = ElevenLabs(api_key="...")
audio = client.generate(
    text="Your text here",
    voice="Rachel",
)
# audio is bytes
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### Coqui TTS (Open Source)

```python
import TTS
tts = TTS.TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to("cuda")
tts.tts_to_file(text="Hello world", file_path="output.wav")
```

### Voice Selection

```python
# List available voices
voices = openai_client.audio.speech.voices()
# Choose by: gender, accent, use case (narration, conversational)
```

### Streaming TTS

```python
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Long text...",
    response_format="mp3"
)
for chunk in response.iter_bytes():
    # Stream to playback
    play_audio_chunk(chunk)
```

---

## Speaker Diarization

**Who spoke when?** Separate audio by speaker without knowing identities.

```python
# pyannote-audio
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipeline("meeting.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

### Diarization + ASR

```python
# 1. Diarize
diarization = diarize_pipeline(audio)
# 2. Transcribe each segment
full_transcript = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    audio_segment = audio[segment.start:segment.end]
    text = asr_model.transcribe(audio_segment)
    full_transcript.append(f"{speaker}: {text}")
```

---

## Voice Cloning and Custom Voices

### Zero-Shot Voice Cloning

Synthesize in a new voice from short reference audio (3–30 seconds).

```python
# ElevenLabs: add custom voice from sample
# OpenAI: limited to preset voices
# Open source: Coqui XTTS, OpenVoice
```

### OpenVoice

```python
# pip install openvoice
from openvoice import OpenVoice
# Clone from reference, control speed and emotion
```

### Fine-Tuning for Voice

- **Few-shot**: Use 1–5 min of target voice
- **Zero-shot**: 3–30 sec reference
- **Fine-tuned**: Hours of data for high fidelity

---

## Voice Assistants and Conversational AI

### Pipeline

1. **Wake word** (optional): "Hey Assistant"
2. **ASR**: Speech → text
3. **NLU/LLM**: Intent, entities, response
4. **TTS**: Response → speech
5. **Playback**

### Turn-Taking

```python
def voice_assistant_loop():
    while True:
        # VAD (Voice Activity Detection) to detect speech start/end
        audio = record_until_silence()
        text = asr.transcribe(audio)
        if not text:
            continue
        response = llm.chat(text)
        audio_response = tts.synthesize(response)
        play(audio_response)
```

### Streaming Conversation

```python
# Stream ASR → stream to LLM → stream TTS
# Reduces latency (start speaking before full input)
```

### Integration with LLM

```python
def voice_agent(user_speech):
    text = asr.transcribe(user_speech)
    response = llm.generate(
        f"User said: {text}\nRespond concisely for speech output (short sentences)."
    )
    return tts.synthesize(response)
```

---

## Speech Emotion Recognition

Classify emotion from speech: happy, sad, angry, neutral, etc.

```python
from transformers import pipeline
pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
result = pipe("speech.wav")
# [{"label": "angry", "score": 0.8}, ...]
```

### Use Cases

- Call center: Detect frustrated customers
- Content moderation: Flag emotional content
- Accessibility: Adapt responses to user state

---

## Practical Examples

### Example 1: Meeting Transcription

```python
def transcribe_meeting(audio_path):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    return result["text"]

# With diarization
def transcribe_meeting_with_speakers(audio_path):
    diarization = diarize_pipeline(audio_path)
    model = whisper.load_model("base")
    transcript = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segment_audio = extract_segment(audio_path, segment)
        text = model.transcribe(segment_audio)["text"]
        transcript.append(f"{speaker}: {text}")
    return "\n".join(transcript)
```

### Example 2: Voice-to-Voice Agent

```python
def voice_agent_loop(asr, llm, tts):
    print("Listening...")
    while True:
        audio = record_from_mic()
        text = asr.transcribe(audio)
        if "goodbye" in text.lower():
            break
        response = llm.chat(text)
        tts.play(response)
```

### Example 3: Podcast Summary

```python
def summarize_podcast(audio_path):
    text = whisper.transcribe(audio_path)["text"]
    summary = llm.generate(f"Summarize this podcast transcript in 3 bullet points:\n{text}")
    return summary
```

### Example 4: Multilingual TTS

```python
# Some TTS supports multiple languages with same voice
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Bonjour, comment allez-vous?"
)
```

---

## Best Practices

1. **Model size vs latency**: base for real-time, large for offline quality
2. **Language**: Specify for better accuracy
3. **Noise**: Use denoising or robust models (Whisper) for noisy audio
4. **TTS**: Match voice to use case (narration vs conversational)
5. **Streaming**: For low latency, use streaming ASR and TTS
6. **Privacy**: Process locally when possible (Whisper can run on-device)

---

## Summary

| Component | Model/Tool | Use Case |
|-----------|------------|----------|
| ASR | Whisper, faster-whisper | Transcription |
| TTS | OpenAI TTS, ElevenLabs, Coqui | Synthesis |
| Diarization | pyannote | Who spoke when |
| Voice clone | OpenVoice, XTTS | Custom voices |
| Emotion | wav2vec2-superb-er | Sentiment |
| Pipeline | ASR → LLM → TTS | Voice agent |

**Libraries**: `whisper`, `faster-whisper`, `pyannote-audio`, `TTS`, `transformers`, `openai`, `elevenlabs`
