from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("dtc_model.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, Age: float = Form(...), Sex: float = Form(...), BP: float = Form(...), 
            Cholesterol: float = Form(...),Na_to_K: float = Form(...)):
    features = np.array([[Age, Sex, BP, Cholesterol, Na_to_K]])
    prediction = model.predict(features)[0]
    return templates.TemplateResponse("form.html", {"request": request, "result": f"Predicción: Clase {prediction}"})
