import urllib.request
import json

# URL da API do Ollama rodando no container ao lado
url = "http://jarvis-ollama:11434/api/generate"

data = {
    "model": "qwen2.5", # Substitua pelo modelo que você baixou
    "prompt": "Olá Jarvis! Qual o status atual dos seus sistemas?",
    "stream": False
}

req = urllib.request.Request(
    url, 
    data=json.dumps(data).encode('utf-8'), 
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print("\n--- Resposta do Jarvis ---")
        print(result.get("response", "Sem resposta"))
        print("--------------------------\n")
except Exception as e:
    print(f"Erro ao conectar com o Ollama: {e}")