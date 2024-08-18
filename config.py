MODEL_PATH = '/data/model/trained_model.pkl'
RAW_BU_FEAT_PATH = '/data/raw/bu_feat.csv.gz'
RAW_TRAIN_DATA_PATH = '/data/raw/train.csv.gz'
RAW_TEST_DATA_PATH = '/data/raw/test.csv.gz'
PROCESSED_TRAIN_DATA_PATH = '/data/processed/train_processed.csv'
PROCESSED_VAL_DATA_PATH = '/data/processed/val_processed.csv'
PROCESSED_TEST_DATA_PATH = '/data/processed/test_processed.csv'
PROCESSED_PREDICTION_PATH = '/data/processed/test_predictions.csv'
FEATURES = ['day_id', 'but_num_business_unit', 'dpt_num_department', 'but_postcode', 'but_latitude',
            'but_longitude', 'but_region_idr_region', 'zod_idr_zone_dgr', 'day_id_week', 'day_id_month',
            'day_id_year']
