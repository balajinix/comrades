def preprocess_data(df, imp_col_list):
    filtered_cols = [col for col in imp_col_list if col in df.columns]
    df = df[filtered_cols]
    threshold = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    return df


def check_null(value_list):
    if not value_list:
        return False
    null_count = sum(1 for v in value_list if v is None)
    return null_count > len(value_list) / 2


def column_value_types(df):
    result = {}
    for col in df.columns:
        types_in_col = df[col].dropna().map(type).unique()
        result[col] = [t.__name__ for t in types_in_col]
    return result