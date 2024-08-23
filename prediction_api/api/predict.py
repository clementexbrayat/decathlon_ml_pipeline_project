import joblib
import psycopg2
import config
import time
import os
from utils import preprocess_calendar_features
from sqlalchemy import create_engine, text
import pickle

# Waiting for the model training to finish
def wait_for_model_training():
    print("Checking for model training completion...")
    while True:
        try:
            engine = create_engine(config.DATABASE_URI)
            with engine.connect() as conn:
                # Check if the trained model table exists and has any entries
                model_exists = conn.execute(
                    text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trained_model');")
                ).scalar()

                if model_exists:
                    # Check if the table contains at least one model entry
                    model_count = conn.execute(
                        text("SELECT COUNT(*) FROM trained_model;")
                    ).scalar()

                    if model_count > 0:
                        print("Model training complete!")
                        return
                print("Waiting for train_model to finish...")
                time.sleep(4)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            time.sleep(4)

# Run the waiting function before starting the app
wait_for_model_training()

# Load the pre-trained model from PostgreSQL
def load_model_from_postgres():
    try:
        engine = create_engine(config.DATABASE_URI)
        with engine.connect() as conn:
            # Fetch the latest model from the trained_model table
            model_blob = conn.execute(
                text("SELECT model_data FROM trained_model ORDER BY id DESC LIMIT 1;")
            ).fetchone()[0]
        
        # Deserialize the model
        model = pickle.loads(model_blob)
        print("Model loaded successfully from PostgreSQL.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from PostgreSQL: {e}")

# Load the model
model = load_model_from_postgres()

def make_prediction(input_df):
    input_df = preprocess_calendar_features(input_df)
    return model.predict(input_df)
