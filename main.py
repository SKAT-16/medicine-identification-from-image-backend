import io
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from typing import List, Dict
from collections import Counter
from PIL import Image

# Load API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini AI SDK
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Structured Prompt
PROMPT = """
You are an advanced AI model specialized in medicine identification.
Given an image of a pill, pill box, or medicine packaging, return details in JSON with two types of identification:
1. **Accurate Identification**: The medicine identification extracted directly from the visible text on the image, such as the name, dosage, or other information on the packaging. This should be considered as the "text-based" identification.
2. **Guessed Identification**: The AI's best guess based on visual recognition and patterns in the image, even if there is no direct text.

Return the details in the following JSON format:

{
  "accurate": {
    "name": "Medicine Name from text",
    "dosage": "Dosage information from text (e.g., 500mg)",
    "side_effects": ["List of common side effects from text (if any)"],
    "manufacturer": "Manufacturer name from text",
    "usage": "Brief description of what this medicine is used for (from text)"
  },
  "guessed": {
    "name": "Medicine Name from AI guess",
    "dosage": "Dosage information from AI guess",
    "side_effects": ["List of common side effects from AI guess"],
    "manufacturer": "Manufacturer name from AI guess",
    "usage": "Brief description of what this medicine is used for (from AI guess)"
  }
}

Ensure the response is **valid JSON** with no extra text. Make sure to clearly differentiate between accurate and guessed identification.
"""

def analyze_image(image: bytes) -> Dict:
    """Send an image to Gemini AI and get structured JSON response."""
    image_obj = Image.open(io.BytesIO(image))

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([PROMPT, image_obj])

    try:
        # Clean the response text (remove the "```json" and "```")
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Parse the cleaned response as JSON
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "Invalid AI response"}

def merge_results(results: List[Dict]) -> Dict:
    """Merge AI results from multiple images into a single medicine profile."""
    merged_data = {
        "accurate": {
            "name": Counter(),
            "dosage": Counter(),
            "side_effects": [],
            "manufacturer": Counter(),
            "usage": Counter(),
        },
        "guessed": {
            "name": Counter(),
            "dosage": Counter(),
            "side_effects": [],
            "manufacturer": Counter(),
            "usage": Counter(),
        },
    }

    for result in results:
        if isinstance(result, dict):  # Check if the result is a dictionary
            # Merge accurate data
            if "accurate" in result:
                accurate_data = result["accurate"]
                if "name" in accurate_data:
                    merged_data["accurate"]["name"][accurate_data["name"]] += 1
                if "dosage" in accurate_data:
                    merged_data["accurate"]["dosage"][accurate_data["dosage"]] += 1
                if "side_effects" in accurate_data:
                    merged_data["accurate"]["side_effects"].extend(accurate_data["side_effects"])
                if "manufacturer" in accurate_data:
                    merged_data["accurate"]["manufacturer"][accurate_data["manufacturer"]] += 1
                if "usage" in accurate_data:
                    merged_data["accurate"]["usage"][accurate_data["usage"]] += 1

            # Merge guessed data
            if "guessed" in result:
                guessed_data = result["guessed"]
                if "name" in guessed_data:
                    merged_data["guessed"]["name"][guessed_data["name"]] += 1
                if "dosage" in guessed_data:
                    merged_data["guessed"]["dosage"][guessed_data["dosage"]] += 1
                if "side_effects" in guessed_data:
                    merged_data["guessed"]["side_effects"].extend(guessed_data["side_effects"])
                if "manufacturer" in guessed_data:
                    merged_data["guessed"]["manufacturer"][guessed_data["manufacturer"]] += 1
                if "usage" in guessed_data:
                    merged_data["guessed"]["usage"][guessed_data["usage"]] += 1

    return {
        "accurate": {
            "name": merged_data["accurate"]["name"].most_common(1)[0][0] if merged_data["accurate"]["name"] else "Unknown",
            "dosage": merged_data["accurate"]["dosage"].most_common(1)[0][0] if merged_data["accurate"]["dosage"] else "Unknown",
            "side_effects": list(set(merged_data["accurate"]["side_effects"])),  # Remove duplicates
            "manufacturer": merged_data["accurate"]["manufacturer"].most_common(1)[0][0] if merged_data["accurate"]["manufacturer"] else "Unknown",
            "usage": merged_data["accurate"]["usage"].most_common(1)[0][0] if merged_data["accurate"]["usage"] else "Unknown",
        },
        "guessed": {
            "name": merged_data["guessed"]["name"].most_common(1)[0][0] if merged_data["guessed"]["name"] else "Unknown",
            "dosage": merged_data["guessed"]["dosage"].most_common(1)[0][0] if merged_data["guessed"]["dosage"] else "Unknown",
            "side_effects": list(set(merged_data["guessed"]["side_effects"])),  # Remove duplicates
            "manufacturer": merged_data["guessed"]["manufacturer"].most_common(1)[0][0] if merged_data["guessed"]["manufacturer"] else "Unknown",
            "usage": merged_data["guessed"]["usage"].most_common(1)[0][0] if merged_data["guessed"]["usage"] else "Unknown",
        },
    }

@app.post("/identify/")
async def identify_medicine(files: List[UploadFile] = File(...)):
    """Processes multiple images of the same medicine and returns a consolidated result."""
    responses = []

    for file in files:
        image_data = await file.read()
        ai_result = analyze_image(image_data)
        
        if isinstance(ai_result, dict) and "error" in ai_result:
            continue  # Skip invalid responses
        responses.append(ai_result)

    if not responses:
        return {"error": "Failed to identify medicine from provided images"}

    final_result = merge_results(responses)
    return {"medicine": final_result}
