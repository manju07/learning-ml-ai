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
10. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
11. [Production Considerations](#production-considerations)
12. [References and Further Reading](#references-and-further-reading)

---

## Introduction to Speech AI

**Speech AI** covers technologies that process and generate human speech: recognition (ASR), synthesis (TTS), understanding, and conversational interfaces. The field has evolved from rule-based systems through Hidden Markov Models (HMMs) and Gaussian Mixture Models (GMMs) to modern self-supervised transformer architectures that learn representations directly from raw waveform.

### Conceptual Foundation: The Speech Processing Pipeline

Speech understanding involves several stages:

1. **Acoustic Modeling**: Map raw audio (waveform or mel-spectrogram) to phonetic units. Modern models learn this implicitly via self-supervision.
2. **Language Modeling**: Resolve ambiguities (e.g., "their" vs "there") using linguistic context—handled by decoder in end-to-end models.
3. **Alignment**: Connect acoustic events to text—critical for subtitles, forced alignment, and streaming.

**Key Intuition**: Transformers in speech (e.g., Whisper, wav2vec2) treat audio as a sequence of frame-level embeddings, much like text transformers treat tokens. Self-supervised pre-training on vast unlabeled audio enables strong zero-shot generalization.

### Key Applications

| Application | Technology | Use Case |
|--------------|------------|----------|
| **Transcription** | ASR | Meetings, podcasts, accessibility |
| **Voice assistants** | ASR + NLU + TTS | Alexa, Siri, custom agents |
| **Voice cloning** | TTS + cloning | Audiobooks, dubbing |
| **Call center AI** | ASR + NLU | Sentiment, summarization |
| **Real-time captioning** | Streaming ASR | Live events |
| **Voice biometrics** | Speaker recognition | Authentication |
| **Low-resource languages** | wav2vec2/HuBERT fine-tuning | Minority languages |

### Pipeline Overview

```
Speech → ASR → Text → NLU/LLM → Response Text → TTS → Speech
         ↑                        ↑
    [VAD optional]          [Streaming optional]
```

---

## Automatic Speech Recognition (ASR)

### Architecture Landscape: CTC vs Seq2Seq vs Hybrid

| Approach | Principle | Pros | Cons |
|----------|-----------|------|------|
| **CTC** (Connectionist Temporal Classification) | Frame-to-label with blank token; collapse repeats | Fast, monotonic alignment, streaming-friendly | Weak language modeling, assumes independence |
| **Seq2Seq** (Attention) | Encoder-decoder with cross-attention | Strong language modeling | Non-monotonic, slower, harder to stream |
| **Whisper** | Encoder-decoder, multilingual | Robust, translates, good on accents | Larger, higher latency |
| **wav2vec2/HuBERT** | Self-supervised encoder + CTC head | Fine-tunable on small data, low-resource | Requires fine-tuning for best results |

### Whisper (OpenAI)

State-of-the-art ASR; multilingual (99 languages); robust to accents and noise. Uses an encoder-decoder transformer trained on 680K hours of weakly labeled web data. The encoder processes 80 mel-filterbank features; the decoder generates text with optional cross-attention over source language.

**Model sizes**: `tiny` (39M) → `base` (74M) → `small` (244M) → `medium` (769M) → `large`/`large-v3` (1550M). Trade-off: latency vs. accuracy.

```python
import whisper
import torch

# Load model (auto-downloads on first use)
# For CPU: use "base" or "small"; for GPU: "medium" or "large"
model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

# Basic transcription
result = model.transcribe("audio.mp3")
print(result["text"])

# Full options with explanations
result = model.transcribe(
    "audio.mp3",
    language="en",           # Specify for better accuracy; None = auto-detect
    task="transcribe",       # "transcribe" (same lang) or "translate" (to English)
    fp16=True,               # Use FP16 on GPU for 2x speed (disable on CPU)
    beam_size=5,             # Beam search width; higher = better quality, slower
    best_of=5,               # Return best of N sampling attempts
    temperature=0.0,         # 0 = greedy/deterministic; higher = more diverse
    condition_on_previous_text=True,  # Use prior segments for context (can drift)
    word_timestamps=True,    # Get per-word timings (useful for subtitles)
    vad_filter=True,         # Skip silent regions (saves compute)
    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
)

# Access segments with timestamps
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
```

### wav2vec2 and HuBERT: Self-Supervised ASR

**wav2vec2** (Facebook AI) learns from raw audio via contrastive learning: mask span of frames, predict which quantized representation belongs to masked region vs. distractors. Enables fine-tuning on 10–100 hours of transcribed data for domain adaptation or low-resource languages.

**HuBERT** (Hidden Unit BERT) uses iterative clustering: cluster frame embeddings → predict cluster IDs for masked frames. Often outperforms wav2vec2 on LibriSpeech with similar data.

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import librosa

# Load pre-trained wav2vec2 (English)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load and preprocess audio: 16kHz mono required
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# Forward pass and decode
with torch.no_grad():
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids)
print(transcription[0])

# Fine-tuning on custom data (conceptual):
# 1. Load processor + model
# 2. Prepare dataset with (audio_path, transcript) pairs
# 3. Train with CTC loss: -sum(log P(correct_label | logits))
# 4. Typically 10-50 epochs on 10-100h of data
```

```python
# HuBERT for ASR: use as feature extractor or fine-tune with CTC head
from transformers import HubertForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

# Same inference pattern as wav2vec2
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
print(processor.batch_decode(pred_ids)[0])
```

### Hugging Face Transformers Pipeline

```python
from transformers import pipeline

# Unified pipeline - supports Whisper, wav2vec2, MMS, etc.
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30,      # Process in 30s chunks (Whisper)
    stride_length_s=5,       # Overlap for continuity
    device=0,               # GPU index; -1 for CPU
)

# Single file
result = pipe("audio.wav")
print(result["text"])

# Batch processing with return_timestamps
result = pipe("audio.wav", return_timestamps="word")
for chunk in result["chunks"]:
    print(f"{chunk['timestamp']}: {chunk['text']}")
```

### Streaming ASR

For real-time: process audio chunks, use streaming models (e.g., faster-whisper, Whisper with chunking). **faster-whisper** uses CTranslate2 (optimized inference) and is 4x faster than original Whisper.

```python
# faster-whisper: CTranslate2 backend, faster inference, lower memory
from faster_whisper import WhisperModel

model = WhisperModel(
    "base",
    device="cuda",           # or "cpu"
    compute_type="float16",  # "int8" for CPU, "float16" for GPU
    download_root="./models" # Cache directory
)

# Transcribe with streaming-like segments
segments, info = model.transcribe(
    "audio.mp3",
    beam_size=5,
    vad_filter=True,         # Voice activity detection
    vad_parameters=dict(min_silence_duration_ms=500),
    word_timestamps=True,
)

for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
    if segment.words:
        for w in segment.words:
            print(f"  {w.word}: {w.start:.2f}-{w.end:.2f}")

# Simulating real-time: process chunks as they arrive
def stream_transcribe(audio_stream, chunk_duration_ms=3000):
    """Process audio in chunks for low-latency streaming."""
    buffer = []
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    for chunk in audio_stream:
        buffer.extend(chunk)
        while len(buffer) >= chunk_samples:
            window = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            segments, _ = model.transcribe(
                (window, sample_rate),
                beam_size=1,  # Faster for streaming
                without_timestamps=True,
            )
            for seg in segments:
                yield seg.text
```

### Timestamps and Word-Level Alignments

Essential for subtitles (SRT/VTT), forced alignment, and speaker-attributed transcripts.

```python
# Whisper with word timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)

for segment in result["segments"]:
    print(f"{segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}")
    if "words" in segment:
        for w in segment["words"]:
            # Word-level timing for precise subtitles
            print(f"  {w['word']}: {w['start']:.2f}s - {w['end']:.2f}s")

# Export to SRT format
def to_srt(segments, filepath):
    with open(filepath, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = format_ts(seg["start"])
            end = format_ts(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")

def format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
```

### Custom Vocabulary / Forced Alignment

For domain terms (names, products, acronyms):
- **Custom vocabulary**: Add terms to decoder vocabulary or use prompt/context (Whisper supports `initial_prompt`).
- **Forced alignment**: Montreal Forced Aligner (MFA) or gentle aligns reference text to audio.
- **Fine-tuning**: Best approach for persistent domain adaptation (medical, legal, brand names).

```python
# Whisper: use initial_prompt to bias toward domain terms
result = model.transcribe(
    "medical_consultation.mp3",
    initial_prompt="Patient presents with hypertension. Medications include lisinopril, amlodipine.",
    # Model will favor these terms when acoustically ambiguous
)

# Montreal Forced Aligner (conceptual - CLI tool)
# mfa align corpus_dir dictionary_path acoustic_model output_dir
```

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

**Who spoke when?** Separate audio by speaker without knowing identities. Combines **speaker embedding** (d-vectors, x-vectors) with **clustering** (agglomerative, spectral) or **segmentation** models. Pipeline: VAD → speaker embedding → clustering → resegmentation.

### pyannote-audio

Industry-standard; requires HuggingFace token acceptance for model access.

```python
from pyannote.audio import Pipeline

# Requires: pip install pyannote.audio
# Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="hf_xxx",  # Or HF_TOKEN env var
)

# Run diarization (returns Annotation object)
diarization = pipeline("meeting.wav", min_duration_on=0.3, min_duration_off=0.5)

# Iterate over labeled segments
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s  {speaker}: ...")

# Access raw segments for merging/smoothing
# diarization is Annotation; segments may be fragmented
```

### Diarization + ASR: Speaker-Attributed Transcription

Combine diarization with ASR for "Speaker A: ... Speaker B: ..." output.

```python
def transcribe_with_speakers(audio_path, diarize_pipeline, asr_model):
    """Produce speaker-attributed transcript."""
    # 1. Diarize
    diarization = diarize_pipeline(audio_path)
    # 2. Load full audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    full_transcript = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Extract segment as numpy array
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_audio = audio[start_sample:end_sample]

        # Transcribe segment (Whisper expects file path or (samples, sr))
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            soundfile.write(f.name, segment_audio, sr)
            text = asr_model.transcribe(f.name)["text"]

        full_transcript.append(f"{speaker}: {text.strip()}")

    return "\n".join(full_transcript)
```

### Common Diarization Parameters

| Parameter | Effect |
|-----------|--------|
| `min_duration_on` | Minimum speech segment length (avoid fragments) |
| `min_duration_off` | Minimum silence between speakers |
| `num_speakers` | Fix speaker count if known (improves accuracy) |
| `max_speakers` | Upper bound when unknown |

---

## Voice Cloning and Custom Voices

### Voice Cloning Modes

| Mode | Reference | Fidelity | Use Case |
|------|-----------|----------|----------|
| **Zero-shot** | 3–30 sec | Good | Quick demos, personal assistants |
| **Few-shot** | 1–5 min | Better | Brand voices, characters |
| **Fine-tuned** | 1+ hours | Highest | Audiobooks, high-profile dubbing |

### Zero-Shot Voice Cloning

Synthesize in a new voice from short reference audio. Encoder extracts speaker embedding; decoder conditions on it.

```python
# Coqui XTTS - open source, multilingual
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text="This speech will match the reference voice's characteristics.",
    file_path="cloned_output.wav",
    speaker_wav="reference_6_to_30_seconds.wav",
    language="en",
)

# Best results: clean reference, 6-30 sec, single speaker, minimal noise
```

### OpenVoice

Instant voice cloning with cross-lingual support; can clone tone and emotion separately.

```python
# pip install openvoice
from openvoice import se_extractor
from openvoice.api import ToneColorConverter, BaseSpeakerTTS

# 1. Extract tone color (speaker embedding) from reference
tone_color = se_extractor.get_se("reference.wav", ToneColorConverter, vad=True)

# 2. Convert base speech to target voice
base_speaker_tts = BaseSpeakerTTS("checkpoints/base_speaker").to("cuda")
tone_color_converter = ToneColorConverter("checkpoints/converter").to("cuda")

# Generate base audio, then convert
# base_speaker_tts.tts("Hello world", "base.wav")
# tone_color_converter.convert("base.wav", "output.wav", tone_color)
```

### Fine-Tuning for Voice

- **LoRA/full fine-tuning**: On TTS model with target voice data (Coqui, Tortoise)
- **Adapter modules**: Lightweight adaptation (e.g., prompt tuning for speech)
- **Voice conversion**: Separate VC model (e.g., So-VITS-SVC) converts any speech to target voice

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

Classify emotion from speech: happy, sad, angry, neutral, fear, disgust, surprise. Models use **wav2vec2/HuBERT** fine-tuned on emotion datasets (RAVDESS, IEMOCAP). SUPERB benchmark provides task-specific fine-tuned models.

### wav2vec2-based Emotion Recognition

```python
from transformers import pipeline

# SUPERB emotion recognition (RAVDESS labels)
pipe = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    top_k=3,  # Return top-3 predictions
)
result = pipe("speech.wav")
# [{"label": "angry", "score": 0.8}, {"label": "neutral", "score": 0.15}, ...]
```

### Continuous (Dimensional) Emotion

Valence-Arousal-Dominance (VAD) models predict continuous scores instead of discrete labels.

```python
# Example: Extract embeddings and train regression head for VAD
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
# Freeze backbone, add regression head for valence, arousal
```

### Use Cases

- **Call center**: Detect frustrated customers, route to human agent
- **Content moderation**: Flag emotional intensity in UGC
- **Accessibility**: Adapt TTS pace/tone to user state
- **Mental health**: Monitor mood from voice over time (with consent)

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

1. **Model size vs latency**: `base` for real-time, `large` for offline quality
2. **Language**: Specify `language` for better accuracy; auto-detect adds latency
3. **Noise**: Use denoising (noisereduce, Demucs) or robust models (Whisper)
4. **TTS**: Match voice to use case (narration vs conversational)
5. **Streaming**: Use streaming ASR and TTS for low latency; pipeline stages in parallel
6. **Privacy**: Process locally when possible (Whisper, wav2vec2 run on-device)
7. **Sample rate**: ASR typically 16kHz; resample if needed with `librosa` or `torchaudio`
8. **VAD**: Use Voice Activity Detection to skip silence and reduce cost

---

## Common Pitfalls and Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| **Empty or garbage transcription** | Wrong sample rate, stereo, corrupt file | Resample to 16kHz mono; check file integrity |
| **Hallucinations in silence** | Whisper extrapolates in quiet regions | Enable `vad_filter=True`; use `vad_parameters` |
| **Wrong language** | Auto-detect confusion | Explicitly set `language` |
| **Slow inference** | CPU, large model | Use `faster-whisper`, GPU, FP16/int8 |
| **TTS sounds robotic** | Low-quality model, wrong settings | Use tts-1-hd; tune stability/similarity (ElevenLabs) |
| **Diarization over-segments** | Too many speakers, short turns | Set `num_speakers` if known; increase `min_duration_on` |
| **Voice clone mismatch** | Noisy/short reference | Use 10–20 sec clean, single-speaker reference |
| **Out-of-memory** | Large model, long audio | Use `tiny`/`base`; process in chunks; reduce `beam_size` |

### Debugging Noisy ASR Output

```python
# Verify audio before ASR
import librosa
audio, sr = librosa.load("audio.wav", sr=None)
print(f"Duration: {len(audio)/sr:.2f}s, SR: {sr}, Channels: {audio.ndim}")
# Resample if needed
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
```

---

## Production Considerations

- **Latency budget**: Voice UX typically <500ms first-token; use streaming, smaller models
- **Cost**: Whisper large is expensive at scale; consider wav2vec2 or cloud ASR APIs
- **Fallbacks**: Have backup ASR (e.g., Google/V AWS) for failures
- **Monitoring**: Log WER (if ground truth), latency percentiles, error rates
- **Rate limiting**: TTS/ASR can be abused; throttle and cache
- **Compliance**: GDPR, consent for recording; redact PII in transcripts

---

## References and Further Reading

- **Whisper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., 2022)
- **wav2vec2**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) (Baevski et al., 2020)
- **HuBERT**: [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) (Hsu et al., 2021)
- **pyannote**: [Speaker diarization documentation](https://github.com/pyannote/pyannote-audio)
- **SUPERB**: [SUPERB: Speech Processing Universal PERformance Benchmark](https://arxiv.org/abs/2105.01051)
- **XTTS**: [Coqui TTS](https://github.com/coqui-ai/TTS)
- **OpenVoice**: [Instant Voice Cloning](https://github.com/myshell-ai/OpenVoice)

---

## Summary

| Component | Model/Tool | Use Case |
|-----------|------------|----------|
| ASR | Whisper, faster-whisper, wav2vec2, HuBERT | Transcription, low-resource |
| TTS | OpenAI TTS, ElevenLabs, Coqui XTTS | Synthesis, cloning |
| Diarization | pyannote | Who spoke when |
| Voice clone | OpenVoice, XTTS | Custom voices |
| Emotion | wav2vec2-superb-er | Sentiment |
| Pipeline | ASR → LLM → TTS | Voice agent |

**Libraries**: `whisper`, `faster-whisper`, `pyannote-audio`, `TTS`, `transformers`, `openai`, `elevenlabs`, `librosa`, `torchaudio`
