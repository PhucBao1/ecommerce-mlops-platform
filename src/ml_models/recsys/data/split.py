def temporal_split(final_df):
    train_cutoff = final_df["purchased_at"].quantile(0.70)

    valid_cutoff = final_df["purchased_at"].quantile(0.85)

    train_df = final_df[final_df["purchased_at"] <= train_cutoff].copy()

    valid_df = final_df[
        (final_df["purchased_at"] > train_cutoff)
        & (final_df["purchased_at"] <= valid_cutoff)
    ].copy()

    test_df = final_df[final_df["purchased_at"] > valid_cutoff].copy()

    return train_df, valid_df, test_df
