import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import config
from utils import preprocess_calendar_features

def process_data():
    # Connect to PostgreSQL database
    engine = create_engine(config.DATABASE_URI)
    
    # Data loading from PostgreSQL
    query_bu_feat = "SELECT * FROM bu_features"
    query_train = "SELECT * FROM train_data"
    query_test = "SELECT * FROM test_data"

    df_bu_feat = pd.read_csv(config.RAW_BU_FEAT_PATH)
    df_train = pd.read_csv(config.RAW_TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.RAW_TEST_DATA_PATH)
    # df_bu_feat = pd.read_sql(query_bu_feat, engine)
    # df_train = pd.read_sql(query_train, engine)
    # df_test = pd.read_sql(query_test, engine)

    # Merging features
    df_train_feat = pd.merge(df_train, df_bu_feat, how="left", on="but_num_business_unit")
    df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on="but_num_business_unit")

    df_train_feat = preprocess_calendar_features(df_train_feat)

    # Split train, val set
    df_train = df_train_feat[df_train_feat.day_id_year < 2017]
    df_val = df_train_feat[df_train_feat.day_id_year == 2017]

    # Save processed data to PostgreSQL
    df_train.to_sql('processed_train_data', engine, if_exists='replace', index=False)
    df_val.to_sql('processed_val_data', engine, if_exists='replace', index=False)
    df_test_feat.to_sql('processed_test_data', engine, if_exists='replace', index=False)

    print("Data processing completed and saved to PostgreSQL.")

if __name__ == "__main__":
    print('Starting data processing')
    process_data()
