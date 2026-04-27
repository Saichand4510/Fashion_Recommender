# Fashion Color Recommender

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **Deployment:** Render  
- **LLM:** Groq (GPT OSS 120B)

---

## 🚀 Approach

1. **User Input**
   - The user provides an image and a query (e.g., outfit preference or occasion).

2. **Color Extraction**
   - The image is preprocessed by:
     - Resizing for efficiency  
     - Cropping the center (based on the assumption that the main subject is centered)  
     - Removing skin tones and background noise using HSV filtering  
   - The processed pixels are then passed to **KMeans clustering** to extract dominant colors.  
   - The dominant colors are converted into **HEX codes**.

3. **LLM Integration**
   - The extracted dominant colors along with the user query are sent to the LLM (**GPT OSS 120B via Groq**).
   - A carefully designed prompt ensures:
     - The model does not assume access to the image  
     - Recommendations are based only on extracted colors  
     - Output is clear, stylist-like, and practical  

4. **Output**
   - The system returns:
     - Dominant colors  
     - Personalized fashion recommendations based on the color palette  

---

## 💡 Key Idea
The system combines **computer vision (color extraction)** with **LLM reasoning** to generate intelligent and context-aware fashion recommendations.




## ⚙️ Setup Instructions

> **Python Version:** Python 3.14

```bash
# 1. Clone the repository
git clone https://github.com/Saichand4510/Fashion_Recommender.git
cd fashionrecommendation

# 2. Create virtual environment (Python 3.14+)
python -m venv venv

# 3. Activate environment

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file and add your API key
# GROQ_API_KEY=your_api_key_here

# 6. Run backend (FastAPI)
uvicorn app:app --reload

# 7. Run frontend (Streamlit) in another terminal
streamlit run frontend.py
