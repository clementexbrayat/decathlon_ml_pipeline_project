# PATH variables
MODEL_PATH = '/data/model/trained_model.pkl'
RAW_BU_FEAT_PATH = '/data/raw/bu_feat.csv.gz'
RAW_TRAIN_DATA_PATH = '/data/raw/train.csv.gz'
RAW_TEST_DATA_PATH = '/data/raw/test.csv.gz'
PROCESSED_TRAIN_DATA_PATH = '/data/processed/train_processed.csv'
PROCESSED_VAL_DATA_PATH = '/data/processed/val_processed.csv'
PROCESSED_TEST_DATA_PATH = '/data/processed/test_processed.csv'
PROCESSED_PREDICTION_PATH = '/data/processed/test_predictions.csv'

# MODEL variables
MODEL_PARAMETERS = {
    'loss': 'squared_error',
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 1.0,
    'criterion': 'friedman_mse',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_depth': 3,
    'min_impurity_decrease': 0.0,
    'init': None,
    'random_state': None,
    'max_features': None,
    'alpha': 0.9,
    'verbose': 0,
    'max_leaf_nodes': None,
    'warm_start': False,
    'validation_fraction': 0.1,
    'n_iter_no_change': None,
    'tol': 1e-4,
    'ccp_alpha': 0.0
}
FEATURES = ['day_id', 'but_num_business_unit', 'dpt_num_department', 'but_postcode', 'but_latitude',
            'but_longitude', 'but_region_idr_region', 'zod_idr_zone_dgr']
