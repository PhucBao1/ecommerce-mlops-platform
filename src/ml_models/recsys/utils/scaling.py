from sklearn.preprocessing import StandardScaler

NUM_COLS = [
    "total_reviews_so_far",
    "avg_price_preference",
    "positive_review_ratio",
    "price",
    "avg_item_sentiment",
]


def scale_features(train_df, valid_df, test_df):

    scaler = StandardScaler()

    train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS])

    valid_df[NUM_COLS] = scaler.transform(valid_df[NUM_COLS])

    test_df[NUM_COLS] = scaler.transform(test_df[NUM_COLS])

    return (train_df, valid_df, test_df, scaler)
