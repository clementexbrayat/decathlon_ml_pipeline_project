import os
# from dotenv import load_dotenv

# Load environment variables from a .env file
# load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'processed_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pkl')
INPUT_DATA_PATH = os.path.join(DATA_DIR, 'input', 'new_data.csv')
OUTPUT_DATA_PATH = os.path.join(DATA_DIR, 'output', 'predictions.csv')








#################################
# Model Parameters
RANDOM_STATE = 42  # Ensures reproducibility by setting a fixed random seed
TEST_SIZE = 0.2  # Percentage of data to be used for testing
MODEL_PARAMS = {
    'n_estimators': 100,  # Number of trees in the forest
    'max_depth': 10,      # Maximum depth of each tree
    'min_samples_split': 2,  # Minimum number of samples required to split an internal node
    'min_samples_leaf': 1,   # Minimum number of samples required to be at a leaf node
    'random_state': RANDOM_STATE
}

# Feature Engineering Parameters
FEATURES = ['day_id', 'but_num_business_unit', 'dpt_num_department', 'but_postcode',
       'but_latitude', 'but_longitude', 'but_region_idr_region',
       'zod_idr_zone_dgr', 'day_id_week', 'day_id_month', 'day_id_year',]  # List of features to be used in the model
TARGET = 'target_column'  # The target variable name

# Hyperparameters
HYPERPARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Logging Configuration
LOG_LEVEL = 'INFO'  # Set to 'DEBUG' for more detailed logs

# Environment Variables (Example usage)
DATABASE_URI = os.getenv('DATABASE_URI', 'default_database_uri')
API_KEY = os.getenv('API_KEY', 'default_api_key')

# Other Parameters
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 10      # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for training

# Monitoring Parameters
MONITORING = {
    'ENABLE': True,
    'METRICS': ['accuracy', 'precision', 'recall'],
    'THRESHOLDS': {
        'accuracy': 0.8,
        'precision': 0.75,
        'recall': 0.7
    }
}