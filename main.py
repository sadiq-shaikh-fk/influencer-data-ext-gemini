import os
import json
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from datetime import datetime
import time
from fastapi import Depends, Header

# ---------- Configure logging ----------
logging.basicConfig(
    format="%(asctime)s | [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ---------- Gemini API Key Management ----------
API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]
current_key_index = 0

# Load prompt files
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
USER_PROMPT_TEMPLATE = load_prompt("prompts/user_prompt.txt")

# ---------- Our API Key ----------
ACCESS_KEY = os.getenv("APP_ACCESS_KEY")

def verify_access(x_api_key: str = Header(...)):
    if x_api_key != ACCESS_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# FastAPI app
app = FastAPI(
    title="Influencer Data Extractor using Gemini 2.0 Flash",
    description="FastAPI wrapper for Gemini 2.0 Flash that parses YouTube channel data into influencer records. Supports multiple API keys for redundancy. Includes error handling and logging.",
    version="1.0.0"
)

# Input model for channel data
class ChannelDataInput(BaseModel):
    channel_id: int
    yt_channel_id: str
    channel_title: str
    channel_desc: str
    channel_publishedat: str
    channel_country: str
    channel_keywords: str
    channel_summary: str


# Rotate to the next API key
def rotate_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)

# Get the current API key
def get_current_key():
    return API_KEYS[current_key_index]


# ---------- Endpoint to generate JSON from channel data ----------
@app.post("/extract-influencer-data", dependencies=[Depends(verify_access)], tags=["Routes"])
def generate_json(input_data: ChannelDataInput):
    global current_key_index
    route = "/extract-influencer-data"
    model_name = "gemini-2.0-flash"

    channel_dict = input_data.model_dump()
    channel_json_block = json.dumps(channel_dict, indent=4)

    user_prompt = f"{USER_PROMPT_TEMPLATE}\n\nChannel Data:\n{channel_json_block}"
    input_size = len(user_prompt.encode("utf-8"))

    for attempt in range(len(API_KEYS)):
        api_key = get_current_key()
        key_info = f"Key #{current_key_index + 1}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "tools": [{"urlContext": {}}],
            "generationConfig": {"responseMimeType": "text/plain"}
        }

        headers = {"Content-Type": "application/json"}

        try:
            start = time.time()
            res = requests.post(url, headers=headers, json=payload)
            duration = round(time.time() - start, 3)

            if res.status_code == 200:
                raw_text = res.json()['candidates'][0]['content']['parts'][0]['text']
                logger.info(f"[{route}] ✅ Success | {key_info} | Input: {input_size} bytes | Duration: {duration}s")
                logger.debug(f"🧠 Gemini Raw Output:\n{raw_text}")

                # Clean ```json ... ``` blocks
                cleaned = re.sub(r"```json|```", "", raw_text).strip()

                try:
                    parsed_json = json.loads(cleaned)
                    return {
                        "output": parsed_json
                        
                    }

                except json.JSONDecodeError as e:
                    logger.error(f"[{route}] ❌ JSON Decode Error | {key_info} | {str(e)}")
                    raise HTTPException(status_code=502, detail="Model responded, but output was not valid JSON.")
            else:
                logger.warning(f"[{route}] ❌ Gemini HTTP {res.status_code} | {key_info} | {res.text.strip()}")
                rotate_key()

        except Exception as e:
            logger.error(f"[{route}] 🔥 Exception | {key_info} | {str(e)}")
            rotate_key()

    logger.critical(f"[{route}] ❌ All API keys failed.")
    raise HTTPException(status_code=503, detail="All API keys exhausted or failed.")

# ---------- Test route endpoint ----------
@app.get("/", dependencies=[Depends(verify_access)], tags=["Routes"])
def root():
    return {"status": "Fame Keeda Gemini API is running", "model": "gemini-2.0-flash"}

