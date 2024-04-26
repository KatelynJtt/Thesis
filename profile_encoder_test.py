import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def profile_categorical_data(df):
    """ 
    Profiles cat columns in panda dataframe to determine if they 
    should be one-hot encoded (nominal) or label encoded (ordinal) 
    """
    profile = {}
    for column in df.select_dtypes(include=['object', 'category', 'int']).columns:
        unique_values = df[column].dropna().unique()
        unique_values_count = len(unique_values)
        # Check if unique values have natural ordering
        if pd.api.types.is_integer_dtype(df[column]):
            # Assume ordinal if integers and range matches count of unique values
            if unique_values_count == df[column].max() - df[column].min() + 1:
                profile[column] = 'ordinal'
            else:
                profile[column] = 'nominal'
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            # Check if column can be converted to a numeric type (simple heuristic for ordinal)
            try:
                pd.to_numeric(df[column], errors='raise')
                profile[column] = 'ordinal'
            except ValueError:
                profile[column] = 'nominal'
        else:
            # Default to nominal if no clear ordinal pattern is detected
            profile[column] = 'nominal'
    return profile


def smart_encode(df, profile):
    le = LabelEncoder()
    """ Encodes columns in dataframe based on the column profile """
    encoder_dict = {}
    for column, col_type in profile.items():
        if col_type == 'nominal':
            encoder = OneHotEncoder()
            transformed = encoder.fit_transform(df[[column]]).toarray()
            # Create a dataframe with encoded columns
            cols = [f"{column}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(transformed, columns=cols)
            df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
            encoder_dict[column] = encoder
        elif col_type == 'ordinal':
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            encoder_dict[column] = encoder
    return df, encoder_dict


# EXAMPLE Use
data = {
    'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
    'Model': ['Accord','Corolla','Focus','A4'],
    'Year': [2008,2012,2011,2010],
}

df = pd.DataFrame(data)
profile = profile_categorical_data(df)
print(profile)

encoded_df, encoder_dict = smart_encode(df, profile)
print(encoded_df)