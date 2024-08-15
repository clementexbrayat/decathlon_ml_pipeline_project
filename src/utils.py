import pickle

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