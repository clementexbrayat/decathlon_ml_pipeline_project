import pandas as pd
import psycopg2
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sqlalchemy import create_engine, text
import pickle
import time
import os
import config
from io import BytesIO

# Save the trained model to PostgreSQL
def save_model_to_postgres(model, engine, table_name="trained_model"):
    try:
        # Serialize the model using pickle
        model_blob = pickle.dumps(model)
        
        # Save the model blob to PostgreSQL
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    model_data BYTEA
                );
            """))
            conn.execute(text(f"""
                INSERT INTO {table_name} (model_data) VALUES (:model_data)
            """), {"model_data": model_blob})
        print(f"Model saved successfully to PostgreSQL table '{table_name}'")
    except Exception as e:
        print(f"Error saving model to PostgreSQL: {e}")
        raise

def train_model():
    # Connect to PostgreSQL database
    engine = create_engine(config.DATABASE_URI)
    
    # Load processed data from PostgreSQL
    df_train = pd.read_sql("SELECT * FROM processed_train_data", engine)
    df_val = pd.read_sql("SELECT * FROM processed_val_data", engine)
    df_test_feat = pd.read_sql("SELECT * FROM processed_test_data", engine)

    y_train = df_train['turnover']
    y_val = df_val['turnover']

    # Define preprocessing pipelines
    num_attrib = ["but_latitude", "but_longitude", "day_id_year"]
    cat_attrib = [
        "day_id_week",
        "day_id_month",
        "but_region_idr_region",
        "zod_idr_zone_dgr",
        "but_num_business_unit",
        "dpt_num_department",
    ]

    # Ensure test data has all necessary columns
    for col in num_attrib + cat_attrib:
        if col not in df_test_feat.columns:
            df_test_feat[col] = 0  # Or use an appropriate default value

    # Preprocessing for numerical data
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

    # Preprocessing for categorical data
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Combine preprocessing pipelines
    preparation_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrib),
        ("cat", cat_pipeline, cat_attrib)
    ])

    # Full pipeline including preprocessing and model
    full_pipeline = Pipeline([
        ('preparation', preparation_pipeline),
        ('model', GradientBoostingRegressor(**config.MODEL_PARAMETERS))
    ])

    # Train the model
    model_final = full_pipeline.fit(df_train, y_train)

    # Predict on validation set
    y_predict_val = model_final.predict(df_val)
    metric_mae = mean_absolute_error(y_val, y_predict_val)
    print(f"Validation MAE: {metric_mae}")

    # Add predictions to validation set
    df_val['prediction'] = y_predict_val

    save_model_to_postgres(model_final, engine)

    # Predict on the test set
    y_pred = model_final.predict(df_test_feat)
    df_test_feat['prediction'] = y_pred

    # Save predictions back to PostgreSQL
    df_test_feat.to_sql('processed_predictions', engine, if_exists='replace', index=False)

    print("Model training completed and saved.")

if __name__ == "__main__":
    # Ensure the data processing step is complete before starting
    while True:
        try:
            engine = create_engine(config.DATABASE_URI)
            with engine.connect() as conn:
                # Check if tables exist in the database
                train_exists = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'processed_train_data');")).scalar()
                val_exists = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'processed_val_data');")).scalar()
                test_exists = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'processed_test_data');")).scalar()

            if all([train_exists, val_exists, test_exists]):
                break
            else:
                print("Waiting for process_data to finish...")
                time.sleep(4)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            time.sleep(4)

    print('Starting model training')
    train_model()
