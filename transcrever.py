import os
import time
from faster_whisper import WhisperModel

# 1. Definir o modelo e o dispositivo de execução
# Opções de modelo: "tiny", "base", "small", "medium"
# Para GTX 1650 / CPU: "small" ou "base" entregam excelente velocidade e boa precisão em PT-BR.
# Para RTX 3060: Pode usar "small" ou "medium".
MODEL_SIZE = "small"

# Define o dispositivo: "cuda" se tiver GPU disponível, ou "cpu"
DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print(f"Loading Faster-Whisper model ({MODEL_SIZE}) on {DEVICE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    start_time = time.time()
    
    # Transcreve o arquivo de áudio forçando o idioma para Português (pt)
    segments, info = model.transcribe(audio_path, language="pt", beam_size=5)

    print(f"\nDetected language: '{info.language}' with probability {info.language_probability:.2f}")
    print("--- Transcrição do Jarvis ---")
    
    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "

    elapsed_time = time.time() - start_time
    print(f"-----------------------------")
    print(f"Tempo de processamento: {elapsed_time:.2f} segundos\n")
    return full_text.strip()

if __name__ == "__main__":
    # Nome do arquivo de teste de áudio depositado na pasta compartilhada
    sample_file = "sample.wav"
    transcribe_audio(sample_file)