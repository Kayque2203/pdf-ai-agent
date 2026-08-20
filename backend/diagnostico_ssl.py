"""
Script de diagnóstico: testa se a conexão com o Google (gRPC e requests)
consegue confiar no certificado da empresa. Rode com: python diagnostico_ssl.py
"""
import os

print("--- Configuração atual ---")
try:
    import certifi
    cert_path = certifi.where()
    print(f"certifi.where(): {cert_path}")
    with open(cert_path, "rb") as f:
        conteudo = f.read()
    print(f"Tamanho do arquivo certifi: {len(conteudo)} bytes")
    print(f"Contém 'Corporativa'? {b'Corporativa' in conteudo}")
except Exception as e:
    print(f"Erro ao checar certifi: {e}")

try:
    import pip_system_certs.wrapt_requests  # noqa: F401
    print("pip_system_certs.wrapt_requests importado com sucesso (patch aplicado).")
except Exception as e:
    print(f"pip_system_certs NAO disponivel: {e}")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

print("\n--- Teste 1: conexao gRPC pura (sem passar cert explicito) ---")
try:
    import grpc
    creds = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("generativelanguage.googleapis.com:443", creds)
    grpc.channel_ready_future(channel).result(timeout=15)
    print("SUCESSO - conexao gRPC autenticou com o certificado da empresa!")
except Exception as e:
    print(f"FALHOU: {e}")

print("\n--- Teste 2: requests para a API do Gemini ---")
try:
    import requests
    r = requests.get("https://generativelanguage.googleapis.com", timeout=15)
    print(f"SUCESSO - requests conectou na API do Gemini, status {r.status_code}")
except Exception as e:
    print(f"FALHOU: {e}")

print("\n--- Teste 3: requests para a Wikipedia ---")
try:
    import requests
    r = requests.get("https://pt.wikipedia.org/wiki/Frank_Burnet", timeout=15)
    print(f"SUCESSO - requests conectou na Wikipedia, status {r.status_code}")
except Exception as e:
    print(f"FALHOU: {e}")
