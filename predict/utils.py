import pickle
from sklearn.base import BaseEstimator, TransformerMixin


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

def load_model(model_path: str):
    """
    Load a trained machine learning model from disk.
    
    :param model_path: Path to the model file.
    :return: Loaded model object.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise