from flask import Blueprint, request, jsonify
from api.predict import make_prediction
from config import FEATURES
import pandas as pd

api_blueprint = Blueprint('api', __name__)

# Define the input data schema
class InputData:
    def __init__(self, day_id, but_num_business_unit, dpt_num_department, but_postcode, but_latitude,
                 but_longitude, but_region_idr_region, zod_idr_zone_dgr, day_id_week, day_id_month,
                 day_id_year):
        self.day_id = day_id
        self.but_num_business_unit = but_num_business_unit
        self.dpt_num_department = dpt_num_department
        self.but_postcode = but_postcode
        self.but_latitude = but_latitude
        self.but_longitude = but_longitude
        self.but_region_idr_region = but_region_idr_region
        self.zod_idr_zone_dgr = zod_idr_zone_dgr
        self.day_id_week = day_id_week
        self.day_id_month = day_id_month
        self.day_id_year = day_id_year

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = InputData(**data)
        input_df = pd.DataFrame([input_data.__dict__])
        input_df = input_df[FEATURES]
        prediction = make_prediction(input_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
