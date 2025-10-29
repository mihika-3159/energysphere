import os, requests
from pathlib import Path

MODEL_PATH = Path("models/transformer_forecast.pt")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "models")

def download_model():
    if MODEL_PATH.exists():
        return
    url = f"{SUPABASE_URL}/storage/v1/object/{MODEL_BUCKET}/transformer_forecast.pt"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(r.content)
        print("Model downloaded successfully.")
    else:
        print(f"Model download failed: {r.status_code}")
