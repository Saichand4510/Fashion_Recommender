from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans

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
Based on the given color palette and user query, suggest matching outfit recommendations.

USER QUERY:
{user_query}

DETECTED COLORS (from the clothing image):
{colors}

INSTRUCTIONS:

- Do NOT assume access to the image
- Use the provided colors as the source of truth
- If clothing type is unclear, make a reasonable assumption (e.g., shirt, dress, etc.)

- Respond in a natural paragraph format (NOT JSON)
- First briefly describe the color palette
- Then give 3 matching recommendations

Each recommendation must include:
- item
- color name + hex code
- reason
- confidence (0 to 1)

Keep the response clear, practical, and stylist-like.
"""


def extract_colors_from_bytes(image_bytes, k=3):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    image = cv2.resize(image, (200, 200))


    h, w, _ = image.shape
    image = image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)


    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)


    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    remove_mask = skin_mask | white_mask | black_mask


    image[remove_mask > 0] = [0, 0, 0]

 
    pixels = image.reshape(-1, 3)

    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(pixels) < k:
        return ["#000000"]

    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

  
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

 
    counts = np.bincount(kmeans.labels_)
    sorted_colors = [
        color for _, color in sorted(zip(counts, colors), reverse=True)
    ]
 
     
    # hex
    hex_colors = [
        '#%02x%02x%02x' % tuple(map(int, c))
        for c in sorted_colors
    ]

    return hex_colors



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

      
        prompt = PROMPT_TEMPLATE.format(
            user_query=user_query,
            colors=", ".join(colors) if colors else "Not provided"
        )

       
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        return {
            "status": "success",
            "colors": colors,
            "recommendation": response.content
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }