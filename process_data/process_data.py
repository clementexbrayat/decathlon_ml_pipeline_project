import pandas as pd
import config

def process_data():
    # Data loading
    df_bu_feat = pd.read_csv(config.RAW_BU_FEAT_PATH)
    df_train = pd.read_csv(config.RAW_TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.RAW_TEST_DATA_PATH)

    # Merging features
    df_train_feat = pd.merge(df_train, df_bu_feat, how="left", on="but_num_business_unit")
    df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on="but_num_business_unit")

    # Split train, val set
    df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
    df_train_feat["day_id_week"] = df_train_feat.day_id.dt.isocalendar().week
    df_train_feat["day_id_month"] = df_train_feat["day_id"].dt.month
    df_train_feat["day_id_year"] = df_train_feat["day_id"].dt.year

    df_train = df_train_feat[df_train_feat.day_id_year < 2017]
    df_val = df_train_feat[df_train_feat.day_id_year == 2017]

    # Save processed data
    df_train.to_csv(config.PROCESSED_TRAIN_DATA_PATH, index=False)
    df_val.to_csv(config.PROCESSED_VAL_DATA_PATH, index=False)
    df_test_feat.to_csv(config.PROCESSED_TEST_DATA_PATH, index=False)

    print("Data processing completed and saved.")

if __name__ == "__main__":
    print('Strating data processing')

    process_data()
