from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans
from rembg import remove
from PIL import Image
import io
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os 
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

PROMPT_TEMPLATE = """
You are an expert fashion stylist AI.

TASK:
Suggest outfit recommendations based on the user query and the detected color palette with their percentage dominance.

USER QUERY:
{user_query}

DETECTED COLORS (from the clothing image):
{colors}

INSTRUCTIONS:

- Do NOT assume access to the image
- The provided colors include their percentage dominance (e.g., #3a3840 (65%), #c4baa5 (35%))
- Treat higher percentage colors as the primary/base color
- Treat lower percentage colors as secondary/accent colors

- If colors are provided:
  - Base your recommendations mainly on the dominant color
  - Use secondary colors for contrast, layering, or accessories

- If colors are NOT provided ("Not provided"):
  - Ignore color palette and respond based only on the user query

- If clothing type is unclear, make a reasonable assumption (e.g., shirt, dress, etc.)

RESPONSE RULES:

- Respond in a natural paragraph format (NOT JSON)

- If colors are provided:
  - First describe the palette mentioning dominant and secondary colors
- If no colors are provided:
  - Skip palette description

- Then give 3 matching recommendations

Each recommendation must include:
- item
- color name followed by HEX code in brackets (e.g., Navy Blue (#1A3D6D))
- reason (must relate to dominant or secondary colors)
- confidence (0 to 1)

- Ensure recommendations feel cohesive with the dominant color
- Use secondary color for contrast or styling balance

Keep the response clear, practical, and stylist-like.
"""


# def extract_colors_from_bytes(image_bytes, k=3):
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#     image = cv2.resize(image, (200, 200))


#     h, w, _ = image.shape
#     image = image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

#     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

#     lower_skin = np.array([0, 20, 70])
#     upper_skin = np.array([20, 255, 255])
#     skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)


#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([180, 30, 255])
#     white_mask = cv2.inRange(hsv, lower_white, upper_white)


#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 50])
#     black_mask = cv2.inRange(hsv, lower_black, upper_black)

#     remove_mask = skin_mask | white_mask | black_mask


#     image[remove_mask > 0] = [0, 0, 0]

 
#     pixels = image.reshape(-1, 3)

#     pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

#     if len(pixels) < k:
#         return ["#000000"]

#     if len(pixels) > 5000:
#         idx = np.random.choice(len(pixels), 5000, replace=False)
#         pixels = pixels[idx]

  
#     kmeans = KMeans(n_clusters=k, n_init=10)
#     kmeans.fit(pixels)

#     colors = kmeans.cluster_centers_

 
#     counts = np.bincount(kmeans.labels_)
#     sorted_colors = [
#         color for _, color in sorted(zip(counts, colors), reverse=True)
#     ]
 
     
#     # hex
#     hex_colors = [
#         '#%02x%02x%02x' % tuple(map(int, c))
#         for c in sorted_colors
#     ]

#     return hex_colors

def extract_colors_from_bytes(image_bytes, k=3):
   
    output = remove(image_bytes)

    image = Image.open(io.BytesIO(output)).convert("RGB")
    image = np.array(image)

    image = cv2.resize(image, (200, 200))
     

    pixels = image.reshape(-1, 3)

  
    pixels = pixels[np.any(pixels > [10, 10, 10], axis=1)]

    if len(pixels) < k:
        return [{"hex": "#000000", "percentage": 100.0}]


    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=k, n_init=8)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    total = counts.sum()


    sorted_data = sorted(zip(counts, colors), reverse=True)


    result = []
    for count, color in sorted_data:
        hex_code = '#%02x%02x%02x' % tuple(map(int, color))
        percent = (count / total) * 100

        result.append({
            "hex": hex_code,
            "percentage": round(percent, 2)
        })
    print(result) 
    return result


@app.post("/recommend")
async def recommend(
    user_query: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        colors = []

    
        if image:
            image_bytes = await image.read()
            colors = extract_colors_from_bytes(image_bytes)

        if colors:    
            colors_text = ", ".join(
            [f"{c['hex']} ({c['percentage']}%)" for c in colors[:2]]
        )
        else:
            colors_text="Not Provided"    
      
        prompt = PROMPT_TEMPLATE.format(
            user_query=user_query,
            colors=colors_text
        )

       
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        return {
            "status": "success",
            "colors": colors[:2],
            "recommendation": response.content
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }