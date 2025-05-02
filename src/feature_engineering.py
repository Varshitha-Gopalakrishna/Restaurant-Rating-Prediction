from sklearn.preprocessing import OrdinalEncoder

def encode_features(df):
    df = df.copy()
    mappings = {}
    categorical_cols = df.select_dtypes(include='object').columns

    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    for i, col in enumerate(categorical_cols):
        mappings[col] = dict(enumerate(encoder.categories_[i]))

    df['rate'] = df['rate'].fillna(df['rate'].mean())
    df['approx_cost_for_2_people'] = df['approx_cost_for_2_people'].fillna(df['approx_cost_for_2_people'].mean())

    return df, mappings
