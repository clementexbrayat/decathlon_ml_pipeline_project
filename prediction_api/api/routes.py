from flask import Blueprint, request, jsonify
from api.predict import make_prediction
import pandas as pd
from datetime import datetime, timedelta

api_blueprint = Blueprint('api', __name__)

# Define the input data schema
class InputData:
    def __init__(self, day_id, but_num_business_unit, dpt_num_department, but_postcode, but_latitude,
                 but_longitude, but_region_idr_region, zod_idr_zone_dgr):
        self.day_id = day_id
        self.but_num_business_unit = but_num_business_unit
        self.dpt_num_department = dpt_num_department
        self.but_postcode = but_postcode
        self.but_latitude = but_latitude
        self.but_longitude = but_longitude
        self.but_region_idr_region = but_region_idr_region
        self.zod_idr_zone_dgr = zod_idr_zone_dgr

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = InputData(**data)
        input_df = pd.DataFrame([input_data.__dict__])
        prediction = make_prediction(input_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@api_blueprint.route('/predict_8_weeks', methods=['POST'])
def predict_8_weeks():
    try:
        data = request.get_json()
        
        # Get the date of the first week for prediction (next saturday)
        today = datetime.today()
        start_date = today + timedelta(days=-today.weekday()+6, weeks=1)

        # Generate data for each day in the next 8 weeks
        rows = []
        for i in range(0, 8):
            new_data = data.copy()
            new_date = start_date + timedelta(days=i*7)
            new_data["day_id"] = new_date.strftime("%Y-%m-%d")
            rows.append(new_data)
        input_df = pd.DataFrame(rows)

        predictions = make_prediction(input_df)
        response = {'predictions': {}}
        for (day_id, prediction) in zip(input_df['day_id'], predictions):
            response['predictions'][str(day_id.strftime("%Y-%m-%d"))] = prediction
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
