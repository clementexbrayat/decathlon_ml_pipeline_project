import joblib
import config
import time
import os
from utils import preprocess_calendar_features

# Waiting for the model training to finish
def wait_for_model_training():
    print("Checking for model training completion...")
    while not os.path.exists(config.MODEL_PATH):
        print("Waiting for train_model to finish...")
        time.sleep(4)
    print("Model training complete!")

# Run the waiting function before starting the app
wait_for_model_training()

# Load the pre-trained model
try:
    model = joblib.load(config.MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError("Model file not found")

def make_prediction(input_df):
    input_df = preprocess_calendar_features(input_df)
    return model.predict(input_df)
