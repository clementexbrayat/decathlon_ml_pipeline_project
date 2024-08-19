# Machine Learning Pipeline Project

This project is a modular machine learning pipeline designed to handle data processing, model training, and prediction serving via an API. Each task is encapsulated in its own Docker container, allowing for easy management, deployment, and retraining of the model as needed.

## Project Structure

The project is organized into three main services:

1. **Data Processor Service**: Extracts and processes raw data from a CSV file.
2. **Model Trainer Service**: Trains a machine learning model using the processed data.
3. **Prediction API Service**: Provides a Flask-based API to serve predictions using the trained model.

### Directory Structure

```plaintext
machine_learning_project/
│
├── data/
│   ├── raw/                      # Directory for raw data
│   │   └── raw_data.csv          # Example raw data file
│   ├── processed/                # Directory for processed data
│   │   └── processed_data.csv    # Example processed data file
│   └── model/                    # Directory for the trained model
│       └── model.pkl             # Trained model file
│
├── data_processor/               # Data Processor Service
│   ├── Dockerfile
│   ├── process_data.py
│   └── requirements.txt
│
├── model_trainer/                # Model Trainer Service
│   ├── Dockerfile
│   ├── train_model.py
│   └── requirements.txt
│
├── prediction_api/               # Prediction API Service
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
│   └── api/
│       └── routes.py
│       └── predict.py
│
├── docker-compose.yml            # Docker Compose configuration file
├── config.py                     # Configuration file
├── utils.py                      # Utility functions file
└── README.md                     # Project README file
```

#### Getting started

- Clone the Repository
https://github.com/clementexbrayat/decathlon_ml_pipeline_project.git
cd decathlon_ml_pipeline_project

- Load your raw data in 'data/raw/' folder

- Build the Docker Images
docker-compose build

- Run the containers
You can eaither run all the pipeline in one go with docker-compose up or run each task individually with docker-compose run tash_name

- Make predictions
Once the Prediction API is running, you can make predictions by sending a POST request to http://localhost:8000/predict with JSON data. Example using curl:
curl -X POST "http://localhost:8000/predict/" 
-H "Content-Type: application/json" 
-d '{"day_id": "2017-11-25", "but_num_business_unit": 95, "dpt_num_department": 73, "but_postcode": 80000, "but_latitude": 49.9, "but_longitude": 2.28, "but_region_idr_region": 69, "zod_idr_zone_dgr": 4}'. 
The API will return a JSON response with the predictions: {"prediction":prediction_value}

