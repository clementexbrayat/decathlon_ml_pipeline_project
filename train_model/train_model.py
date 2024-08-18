import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import time
import os
import config

def train_model():
    # Load processed data
    df_train = pd.read_csv(config.PROCESSED_TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.PROCESSED_VAL_DATA_PATH)
    df_test_feat = pd.read_csv(config.PROCESSED_TEST_DATA_PATH)

    y_train = df_train.turnover
    y_val = df_val.turnover

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
        ('model', GradientBoostingRegressor())
    ])

    # Train the model
    model_final = full_pipeline.fit(df_train, y_train)

    # Predict on validation set
    y_predict_val = model_final.predict(df_val)
    metric_mae = mean_absolute_error(y_val, y_predict_val)
    print(f"Validation MAE: {metric_mae}")

    # Add predictions to validation set
    df_val['prediction'] = y_predict_val

    # Save the trained model
    def save_model(model, model_path: str):
        try:
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    save_model(model_final, config.MODEL_PATH)

    # Predict on the test set
    y_pred = model_final.predict(df_test_feat)
    df_test_feat['prediction'] = y_pred
    df_test_feat.to_csv(config.PROCESSED_PREDICTION_PATH, index=False)

    print("Model training completed and saved.")

if __name__ == "__main__":
    while not all(os.path.exists(file) for file in [config.PROCESSED_TRAIN_DATA_PATH, config.PROCESSED_VAL_DATA_PATH, config.PROCESSED_TEST_DATA_PATH]):
        print("Waiting for process_data to finish...")
        time.sleep(4)
    
    print('Strating model training')

    train_model()
