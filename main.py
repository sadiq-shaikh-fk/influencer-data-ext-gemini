import os
import json
import re
import requests
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import time

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s | [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------- Load env var ----------
load_dotenv()

API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]
current_key_index = 0
ACCESS_KEY = os.getenv("APP_ACCESS_KEY")

# ---------- Load prompt files ----------
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
USER_PROMPT_TEMPLATE = load_prompt("prompts/user_prompt.txt")

# ---------- Auth ----------
def verify_access(x_api_key: str = Header(...)):
    if x_api_key != ACCESS_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# ---------- FastAPI ----------
app = FastAPI(
    title="Influencer Data Extractor using Gemini 2.0 Flash",
    description="Converts YouTube channel data into influencer records using Gemini 2.0 Flash with retry and merge logic for gender and city.",
    version="1.0.0"
)

# ---------- Input Schema ----------
class ChannelDataInput(BaseModel):
    channel_id: int
    yt_channel_id: str
    channel_title: str
    channel_desc: str
    channel_publishedat: str
    channel_country: str
    channel_keywords: str
    channel_summary: str

# ---------- API Key logic ----------
def rotate_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)

def get_current_key():
    return API_KEYS[current_key_index]

# ---------- Main Endpoint ----------
@app.post("/extract-influencer-data", dependencies=[Depends(verify_access)], tags=["Routes"])
def generate_json(input_data: ChannelDataInput):
    global current_key_index
    route = "/extract-influencer-data"
    model_name = "gemini-2.0-flash"

    channel_dict = input_data.model_dump()
    channel_json_block = json.dumps(channel_dict, indent=4)
    input_size = len(channel_json_block.encode("utf-8"))

    user_prompt_base = f"{USER_PROMPT_TEMPLATE}\n\nChannel Data:\n{channel_json_block}"
    merged_result = {}
    best_score = -1

    for attempt in range(1, 4):
        api_key = get_current_key()
        key_info = f"Key #{current_key_index + 1}"

        is_retry = attempt > 1
        prompt = user_prompt_base
        if is_retry:
            prompt += (
                "\n\nRetry reasoning and try to infer gender and city if possible. "
                "You may guess gender based on the name (e.g., 'Pallavi' is usually Female). "
                "For city, infer from description, language, country, or context if possible. "
                "Return only valid JSON."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "tools": [{"urlContext": {}}],  # Optional URL context
            "generationConfig": {"responseMimeType": "text/plain"}
        }

        headers = {"Content-Type": "application/json"}

        try:
            logger.info(f"[{route}] 🤖 Attempt #{attempt} using {key_info}")
            start = time.time()
            res = requests.post(url, headers=headers, json=payload)
            duration = round(time.time() - start, 2)

            if res.status_code != 200:
                logger.warning(f"[{route}] ❌ Gemini HTTP {res.status_code} | {key_info} | {res.text.strip()}")
                rotate_key()
                continue

            raw_text = res.json()['candidates'][0]['content']['parts'][0]['text']
            cleaned = re.sub(r"```json|```", "", raw_text).strip()
            parsed = json.loads(cleaned)[0]

            gender_ok = parsed.get("inf_gender") not in (None, "", "null")
            city_ok = parsed.get("inf_city") not in (None, "", "null")
            score = int(gender_ok) + int(city_ok)

            logger.info(f"[{route}] ✅ Attempt #{attempt} | gender: {gender_ok} | city: {city_ok} | Score: {score} | Time: {duration}s")

            if score > best_score:
                merged_result = parsed
                best_score = score

            if best_score == 2:
                break

        except Exception as e:
            logger.error(f"[{route}] 🔥 Exception on attempt #{attempt} | {key_info} | {str(e)}")
            rotate_key()

    if not merged_result:
        raise HTTPException(status_code=502, detail="Model did not return valid response after retries.")

    # Add country to result
    merged_result["channel_country"] = input_data.channel_country

    return {
        "json_result": [merged_result]
    }

# ---------- Health Check ----------
@app.get("/", dependencies=[Depends(verify_access)], tags=["Routes"])
def root():
    return {"status": "Fame Keeda Gemini API is running", "model": "gemini-2.0-flash"}
