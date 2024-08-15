import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Data loading
df_bu_feat = pd.read_csv("../data/raw/bu_feat.csv.gz")
df_train = pd.read_csv("../data/raw/train.csv.gz")
df_test = pd.read_csv("../data/raw/test.csv.gz")

# Merging features
df_train_feat = pd.merge(df_train, df_bu_feat, how="left", on="but_num_business_unit")
df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on="but_num_business_unit")

# Split train, val set
df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
df_train_feat["day_id_week"] = df_train_feat.day_id.dt.isocalendar().week
df_train_feat["day_id_month"] = df_train_feat["day_id"].dt.month
df_train_feat["day_id_year"] = df_train_feat["day_id"].dt.year

df_train = df_train_feat[df_train_feat.day_id_year < 2017]
df_val = df_train_feat[df_train_feat.day_id_year == 2017]

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

# Output unique values for validation check
print(df_val['dpt_num_department'].unique())

# Add predictions to validation set
df_val['prediction'] = y_predict_val

# Save the trained model
def save_model(model, model_path: str):
    """
    Save the trained machine learning model to disk.
    
    :param model: Trained model object.
    :param model_path: Path to save the model file.
    """
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

save_model(model_final, 'trained_model_2.pkl')

# Predict on the test set
y_pred = model_final.predict(df_test_feat)
df_test_feat['prediction'] = y_pred
