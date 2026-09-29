import json
import os
from PIL import Image
from google import genai

def analyze_site_image(image_path_or_pil, api_key: str) -> dict:
    """
    Analyzes telecom site photos using Google Gemini Vision API
    and returns parsed inspection JSON metrics.
    """
    try:
        client = genai.Client(api_key=api_key)
        
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil

        prompt = """
        Analyze this telecom site equipment/shelter photo. 
        Detect issues like corrosion, oil leaks, battery swelling, missing bolts, cable damage, and poor housekeeping.
        
        Return ONLY a raw JSON object with no markdown fences, formatted exactly as follows:
        {
            "corrosion_score": <int 0-100 where 0 is none and 100 is severe>,
            "battery_swelling": <boolean true/false>,
            "oil_leakage": <boolean true/false>,
            "missing_bolts": <boolean true/false>,
            "cable_damage": <boolean true/false>,
            "housekeeping_score": <int 0-100 where 100 is pristine and 0 is hazardous>,
            "summary_notes": "<string brief explanation of findings>"
        }
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt]
        )
        
        text_response = response.text.strip()
        if text_response.startswith("```json"):
            text_response = text_response.replace("```json", "").replace("```", "").strip()
            
        return json.loads(text_response)

    except Exception as e:
        return {
            "error": str(e),
            "corrosion_score": 0,
            "battery_swelling": False,
            "oil_leakage": False,
            "missing_bolts": False,
            "cable_damage": False,
            "housekeeping_score": 50,
            "summary_notes": f"Analysis failed: {str(e)}"
        }
