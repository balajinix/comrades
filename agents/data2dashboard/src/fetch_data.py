def get_data_schema(df):
    schema = {}
    for col in df.columns:
        types = df[col].dropna().map(lambda x: type(x).__name__).unique().tolist()
        schema[col] = types
    return schema

def extract_data(df, col_name):
    if col_name in df.columns:
        return df[col_name].tolist()
    else:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")


def filter_csv_by_col(data_schema, imp_col_list):
    return {key: value for key, value in data_schema.items() if key in imp_col_list}

def keep_important_columns(df, important_columns):
    columns_to_keep = [col for col in important_columns if col in df.columns]
    return df[columns_to_keep].copy()