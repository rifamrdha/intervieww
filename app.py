from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import os
import json

# ==========================================
# TOKEN (OPSIONAL, AMAN UNTUK RAILWAY)
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Jika ada token, pakai. Kalau tidak ada, pakai public inference.
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ==========================================
# MODEL CONFIG
# ==========================================
WHISPER_MODEL = "openai/whisper-large-v2"   # Transkripsi
QWEN_MODEL = "Qwen/Qwen2-1.5B-Instruct:featherless-ai"  # Evaluasi
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

app = FastAPI(title="AI Interview Backend (Whisper + Qwen)")


# ==========================================
# ENDPOINT: TRANSCRIBE (Whisper)
# ==========================================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        whisper_url = f"https://api-inference.huggingface.co/models/{WHISPER_MODEL}"

        resp = requests.post(
            whisper_url,
            headers=HEADERS,
            data=audio_bytes
        )

        if resp.status_code != 200:
            raise HTTPException(500, resp.text)

        return resp.json()
    except Exception as e:
        raise HTTPException(500, str(e))


# ==========================================
# ENDPOINT: EVALUATE (Qwen)
# ==========================================
@app.post("/evaluate")
async def evaluate_answer(
    filename: str,
    question_text: str,
    clean_transcript: str,
    rubric_for_video: dict
):
    # Bangun payload prompt untuk Qwen
    messages = [
        {"role": "system", "content": "You are an interview evaluator."},
        {
            "role": "user",
            "content": json.dumps({
                "filename": filename,
                "question": question_text,
                "answer": clean_transcript,
                "rubric": rubric_for_video
            }, ensure_ascii=False)
        }
    ]

    payload = {"model": QWEN_MODEL, "messages": messages}

    try:
        resp = requests.post(
            CHAT_URL,
            headers={**HEADERS, "Content-Type": "application/json"},
            json=payload
        )

        if resp.status_code != 200:
            raise HTTPException(500, resp.text)

        return resp.json()
    except Exception as e:
        raise HTTPException(500, str(e))


# ==========================================
# OPTIONAL: ROOT CHECKER
# ==========================================
@app.get("/")
def root():
    return {
        "status": "running",
        "endpoints": ["/transcribe", "/evaluate", "/docs"]
    }
