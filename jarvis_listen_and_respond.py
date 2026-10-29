import os
import json
import urllib.request
from faster_whisper import WhisperModel

# 1. Transcrever o Áudio de Entrada
print("1. Ouvindo o arquivo de áudio...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
audio_file = "/opt/jarvis/workspace/sample.wav"

segments, _ = whisper_model.transcribe(audio_file, language="pt")
prompt_text = " ".join([segment.text for segment in segments]).strip()

print(f"Texto Transcrito: '{prompt_text}'")

# 2. Enviar o Texto ao Cérebro (Ollama)
print("\n2. Processando resposta no Ollama...")
ollama_url = "http://jarvis-ollama:11434/api/generate"

payload = {
    "model": "qwen2.5:3b",  # Alterne para o modelo baixado no seu container
    "prompt": f"Você é o Jarvis, um assistente virtual atencioso. Responda de forma sucinta ao comando: {prompt_text}",
    "stream": False
}

req = urllib.request.Request(
    ollama_url,
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print("\n=== Resposta do Jarvis ===")
        print(result.get("response", ""))
        print("==========================")
except Exception as e:
    print(f"Erro ao conectar com Ollama: {e}")