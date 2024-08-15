from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List
from config import MODEL_PATH, FEATURES

# from utils import CustomPreprocressing


# Initialize FastAPI app
app = FastAPI()


# Load the pre-trained model
try:
    print("Loading model...")
    # model = joblib.load(MODEL_PATH)
    model = joblib.load('trained_model.pkl')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found")

# Define the input data model
class InputData(BaseModel):
    day_id: str
    but_num_business_unit: int
    dpt_num_department: int
    but_postcode: int
    but_latitude: float
    but_longitude: float
    but_region_idr_region: int
    zod_idr_zone_dgr: int
    day_id_week: int
    day_id_month: int
    day_id_year: int
    # Add other features as necessary

# Define the output data model
class Prediction(BaseModel):
    prediction: float

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Machine Learning Prediction API"}

# Define the prediction endpoint
@app.post("/predict/", response_model=Prediction)
def predict(data: InputData):
    try:
        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Ensure the DataFrame has the correct feature columns
        input_df = input_df[FEATURES]
        
        # Make predictions using the pre-trained model
        prediction = model.predict(input_df)
        
        # Return the prediction as a JSON response
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Start the application (for local development; in production, use a WSGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
