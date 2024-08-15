
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

import pickle



# ## Data loading


df_bu_feat = pd.read_csv("../data/raw/bu_feat.csv.gz") 
df_train = pd.read_csv("../data/raw/train.csv.gz") 
df_test = pd.read_csv("../data/raw/test.csv.gz") 


# ### Merging features


df_train_feat = pd.merge(df_train, df_bu_feat, how="left", on = "but_num_business_unit")
df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on = "but_num_business_unit")


# ### Split train, val set


# Train and val set

df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
df_train_feat["day_id_week"] = df_train_feat.day_id.dt.isocalendar().week
df_train_feat["day_id_month"] = df_train_feat["day_id"].dt.month
df_train_feat["day_id_year"] = df_train_feat["day_id"].dt.year

df_train = df_train_feat[(df_train_feat.day_id_year < 2017)]
df_val = df_train_feat[(df_train_feat.day_id_year == 2017)]

y_train = df_train.turnover
y_val = df_val.turnover


# ### Scikit pipeline

class CustomPreprocressing(BaseEstimator, TransformerMixin):
    """
    This class includes all the steps for the preprocessing
    """
    def __init__(self, cat_cols):
        """
        Initialize the class / Can be empty
        """
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        """
        This method is only created so that the pipeline containing this transformer does not raise an error
        """
        return self

    def transform(self, data):
        """
        Inputs :
          -- data : DataFrame, DataFrame contening all the data needed for the model
        Outputs :
          -- DataFrame, DataFrame prepared for modeling

        """
        data["day_id"] = pd.to_datetime(data["day_id"])
        data["day_id_week"] = data.day_id.dt.isocalendar().week
        data["day_id_month"] = data["day_id"].dt.month
        data["day_id_year"] = data["day_id"].dt.year
        data[self.cat_cols] = data[self.cat_cols].apply(lambda x: x.astype(str))
        return data



num_attrib = ["but_latitude","but_longitude", 'day_id_year']
cat_attrib = [
            "day_id_week",
            "day_id_month",
            "but_region_idr_region",
            "zod_idr_zone_dgr",
            "but_num_business_unit",
            "dpt_num_department",
        ]

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
cat_onehot_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown="ignore")),
])
preparation_pipeline = ColumnTransformer([
    ("num",num_pipeline, num_attrib),
    ("cat_onehot", cat_onehot_pipeline, cat_attrib)
])

full_pipeline = Pipeline([
    ('preprocessing', CustomPreprocressing(cat_cols=cat_attrib )),
    ('preparation', preparation_pipeline),
    ('model', GradientBoostingRegressor())
])





model_final = full_pipeline.fit(df_train, y_train)


model_final = full_pipeline.fit(df_train, y_train)
y_predict_val = model_final.predict(df_val)

metric_mae = mean_absolute_error(y_val, y_predict_val)


df_val['dpt_num_department'].unique()


df_val['prediction'] = y_predict_val
filter_ts = lambda x: (x.but_num_business_unit=="32") & (x.dpt_num_department=='73')


model_final = full_pipeline.fit(df_train_feat, df_train_feat.turnover.values)

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


y_pred = model_final.predict(df_test_feat)


df_test_feat['prediction'] = y_pred
