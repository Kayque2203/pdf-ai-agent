"""
Script de diagnóstico: testa se a conexão gRPC com o Google consegue
confiar no certificado da empresa, isolado de toda a complexidade do
LangChain. Rode com: python diagnostico_ssl.py
"""
import os
from pathlib import Path

CERT_PATH = Path(__file__).resolve().parent / "combined_cert.pem"

print(f"Procurando certificado em: {CERT_PATH}")
print(f"Arquivo existe? {CERT_PATH.exists()}")
if CERT_PATH.exists():
    print(f"Tamanho do arquivo: {CERT_PATH.stat().st_size} bytes")

os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = str(CERT_PATH)
os.environ["SSL_CERT_FILE"] = str(CERT_PATH)
os.environ["REQUESTS_CA_BUNDLE"] = str(CERT_PATH)

print("\n--- Teste 1: conexão gRPC pura (sem passar cert explícito) ---")
try:
    import grpc
    creds = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("generativelanguage.googleapis.com:443", creds)
    grpc.channel_ready_future(channel).result(timeout=15)
    print("✅ SUCESSO - conexão gRPC autenticou com o certificado da empresa!")
except Exception as e:
    print(f"❌ FALHOU: {e}")

print("\n--- Teste 2: conexão gRPC passando o certificado manualmente ---")
try:
    import grpc
    with open(CERT_PATH, "rb") as f:
        cert_bytes = f.read()
    creds2 = grpc.ssl_channel_credentials(root_certificates=cert_bytes)
    channel2 = grpc.secure_channel("generativelanguage.googleapis.com:443", creds2)
    grpc.channel_ready_future(channel2).result(timeout=15)
    print("✅ SUCESSO - funcionou passando o certificado manualmente!")
except Exception as e:
    print(f"❌ FALHOU: {e}")

print("\n--- Teste 3: requests (HTTP comum, não gRPC) ---")
try:
    import requests
    r = requests.get("https://generativelanguage.googleapis.com", timeout=15)
    print(f"✅ SUCESSO - requests conectou, status {r.status_code}")
except Exception as e:
    print(f"❌ FALHOU: {e}")
