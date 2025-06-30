import os
import json
import re
import requests
import pandas as pd
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

# ---------- Load CSV data at startup ----------
def load_cities_data():
    try:
        cities_df = pd.read_csv("data/cities.csv")
        logger.info(f"✅ Loaded {len(cities_df)} cities from dataset")
        return cities_df
    except FileNotFoundError:
        logger.error("❌ cities.csv not found in data/ directory")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Error loading cities.csv: {str(e)}")
        return pd.DataFrame()

# Load cities data globally
CITIES_DATA = load_cities_data()

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

# ---------- Prepare cities context for prompt ----------
def prepare_cities_context(channel_country: str) -> str:
    if CITIES_DATA.empty:
        return "No cities dataset available."
    
    # Filter cities by country if possible
    country_cities = CITIES_DATA[CITIES_DATA['countryCode'] == channel_country]
    
    if len(country_cities) == 0:
        # If no cities found for specific country, use top 100 cities globally
        sample_cities = CITIES_DATA.nlargest(100, 'population')
    else:
        # Use cities from the specific country + top global cities for context
        top_global = CITIES_DATA.nlargest(50, 'population')
        sample_cities = pd.concat([country_cities, top_global]).drop_duplicates()
    
    # Convert to compact format for prompt
    cities_list = []
    for _, row in sample_cities.head(200).iterrows():  # Limit to prevent prompt bloat
        cities_list.append(f"{row['cityName']}, {row['countryName']} ({row['countryCode']})")
    
    cities_context = f"""
AVAILABLE CITIES DATASET REFERENCE (Sample of {len(sample_cities)} cities):
{'; '.join(cities_list)}

VALIDATION RULES:
- City predictions MUST match cities from this dataset
- Prioritize cities with countryCode matching channel_country: {channel_country}
- Use exact cityName spelling from dataset
- If no clear match found in dataset, set inf_city to null
"""
    return cities_context

# ---------- Main Endpoint ----------
@app.post("/extract-influencer-data", dependencies=[Depends(verify_access)], tags=["Routes"])
def generate_json(input_data: ChannelDataInput):
    global current_key_index
    route = "/extract-influencer-data"
    model_name = "gemini-2.0-flash"

    channel_dict = input_data.model_dump()
    channel_json_block = json.dumps(channel_dict, indent=4)
    
    # Add cities context to the prompt
    cities_context = prepare_cities_context(input_data.channel_country)
    
    user_prompt_base = f"{USER_PROMPT_TEMPLATE}\n\n{cities_context}\n\nChannel Data:\n{channel_json_block}"
    merged_result = {}
    best_score = -1
    total_duration = 0.0

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
                "IMPORTANT: Use a city from the provided dataset if a good match exists. "
                "BUT if your best guess city is not found in the dataset, still include it in the output and mark it as unverified (add 'inf_city_verified': false in the JSON). "
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
            total_duration += duration

            if res.status_code != 200:
                logger.warning(f"[{route}] ❌ Gemini HTTP {res.status_code} | {key_info} | {res.text.strip()}")
                rotate_key()
                continue

            raw_text = res.json()['candidates'][0]['content']['parts'][0]['text']
            cleaned = re.sub(r"```json|```", "", raw_text).strip()
            parsed = json.loads(cleaned)[0]
            # Fallback: Ensure ig_username exists and is a string
            if "ig_username" not in parsed or parsed["ig_username"] is None:
                parsed["ig_username"] = ""
            elif parsed["ig_username"].startswith("@"):
                parsed["ig_username"] = parsed["ig_username"][1:]

            # Validate city against dataset
            predicted_city = parsed.get("inf_city")
            if predicted_city and not CITIES_DATA.empty:
                city_exists = CITIES_DATA['cityName'].str.lower().eq(predicted_city.lower()).any()
                if not city_exists:
                    logger.warning(f"[{route}] ⚠️ Predicted city '{predicted_city}' not found in dataset, but returning it as unverified.")
                    parsed["inf_city_verified"] = False  # Optional extra metadata
                else:
                    parsed["inf_city_verified"] = True  # Optional: mark it verified

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

    # display total duration and attempts
    logger.info(f"[{route}] 🟢 Completed in {round(total_duration, 2)}s across {attempt} attempt(s)")

    return {
        "json_result": [merged_result]
    }

# ---------- Health Check ----------
@app.get("/", dependencies=[Depends(verify_access)], tags=["Routes"])
def root():
    cities_count = len(CITIES_DATA) if not CITIES_DATA.empty else 0
    return {
        "status": "Fame Keeda Gemini API is running", 
        "model": "gemini-2.0-flash",
        "cities_loaded": cities_count
    }

# ---------- Optional: Endpoint to check cities data ----------
@app.get("/cities/stats", dependencies=[Depends(verify_access)], tags=["Debug"])
def cities_stats():
    if CITIES_DATA.empty:
        return {"error": "No cities data loaded"}
    
    return {
        "total_cities": len(CITIES_DATA),
        "countries": CITIES_DATA['countryCode'].nunique(),
        "sample_cities": CITIES_DATA.head(5)[['cityName', 'countryName', 'countryCode']].to_dict('records')
    }