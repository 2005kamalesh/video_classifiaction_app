from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class VideoData(BaseModel):
    title: str
    description: str

# Load your models and vectorizers
with open("tech_nontech_model.pkl", "rb") as model_file:
    tech_model = pickle.load(model_file)

with open("tech_nontech_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open("domain_model.pkl", "rb") as domain_model_file:
    domain_model = pickle.load(domain_model_file)

with open("domain_vectorizer.pkl", "rb") as domain_vectorizer_file:
    domain_vectorizer = pickle.load(domain_vectorizer_file)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Video Classification API"}

@app.post("/predict_tech/")
async def predict_tech(video: VideoData):
    try:
        combined_text = f"{video.title} {video.description}"
        vectorized_input = vectorizer.transform([combined_text])
        prediction = tech_model.predict(vectorized_input)[0]
        result = "Technical-related" if prediction == 1 else "Not technical-related"
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_domain/")
async def predict_domain(video: VideoData):
    try:
        combined_text = f"{video.title} {video.description}"
        vectorized_input = domain_vectorizer.transform([combined_text])
        domain_prediction = domain_model.predict(vectorized_input)[0]
        domain_mapping = {
            0: "AI",
            1: "Web Development",
            2: "Data Science",
            3: "Cybersecurity",
            4: "Cloud Computing",
            5: "Mobile Development",
            6: "DevOps",
        }
        domain_name = domain_mapping.get(domain_prediction, "Unknown Domain")
        return {"domain": domain_name}
    except Exception as e:
        return {"error": str(e)}
