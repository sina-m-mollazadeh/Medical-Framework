import joblib
import cloudpickle
import pandas as pd
import numpy as np
def is_pure_date_string(s):
    try:
        s = s.strip()
        if any(char.isalpha() for char in s):
            return False
        Sep=""
        if("/" in s):
            Sep=s.split("/")
        elif("-" in s):
            Sep=s.split("-")
        else:
            return False
        if(len(Sep)==3):
            try:
                _,_,_=map(int,Sep)
                return True
            except:
                return False
    except:
        return False


def identify_date_columns(data):
    """
    Detect columns where all non-null values can be successfully converted to datetime.
    """
    date_columns = []

    for col in data.select_dtypes(include=['object', 'string']):
        non_null_values = data[col].dropna()

        try:
            pd.to_datetime(non_null_values, errors='raise', infer_datetime_format=True)
            date_columns.append(col)
        except Exception:
            continue

    return date_columns

def convert_jalali_column_to_datetime(col):
    converted = []
    for val in col:
        try:
            val = str(val).strip()
            parts = val.split('/')
            if len(parts) == 3:
                year, month, day = map(int, parts)
                jdate = jdatetime.date(year, month, day).togregorian()
                dt = pd.Timestamp(jdate)
            else:
                dt = pd.NaT
        except:
            dt = pd.NaT
        converted.append(dt)
    return pd.Series(converted)

def detect_calendar_type(date_str):
    """
    Detects whether a date is likely Jalali or Gregorian based on year.
    Returns 'jalali', 'gregorian', or 'unknown'.
    """
    try:
        date_str = str(date_str).strip()
        parts=""
        if("/" in date_str):
            parts = date_str.split('/')
        elif("-" in date_str):
            parts = date_str.split('/')
        if len(parts) != 3:
            return 'unknown'
        year = int(parts[0])
        if 1200 <= year <= 1500:
            return 'jalali'
        elif 1800 <= year <= 2200:
            return 'gregorian'
        else:
            return 'unknown'
    except:
        return 'unknown'

def process_date_columnsExp(data, reference_column=None):
    """ Processes the date columns and extracts useful time-based features."""

    date_columns=identify_date_columns(data)

    if(len(date_columns)==0):
        return data
    typeCal=detect_calendar_type((data[date_columns[0]].values)[0])
    if(typeCal=="jalali"):
        for col in date_columns:
            data[col] = convert_jalali_column_to_datetime(data[col])
    else:
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    for col in date_columns:
        data[f'{col}_year'] = data[col].dt.year.astype(float)
        data[f'{col}_month'] = data[col].dt.month.astype(float)
        data[f'{col}_day'] = data[col].dt.day.astype(float)
        data[f'{col}_day_of_week'] = data[col].dt.dayofweek.astype(float)
        data[f'{col}_hour'] = data[col].dt.hour.astype(float)
        data[f'{col}_day_of_year'] = data[col].dt.dayofyear.astype(float)
        data[f'{col}_is_weekend'] = (data[col].dt.weekday >= 5).astype(float)


    if reference_column:
        for col in date_columns:
            data[f'{col}_time_diff'] = (data[reference_column] - data[col]).dt.total_seconds() / (60 * 60 * 24)

    return data.drop(columns=date_columns)

X_test=pd.read_csv("./test_data.csv")
all_mappings = joblib.load("all_mapping.pkl")
selected_columns = joblib.load("features.pkl")

X_test=process_date_columnsExp(X_test)
for name in X_test.columns:
    if(abs(len(X_test)-X_test[name].isna().sum()) < 3 ):
        X_test.drop(columns=name,inplace=True)

for column, mapping in all_mappings.items():
    if column in X_test.columns:
        X_test[column] = X_test[column].map(mapping)

with open("class_mean_imputer.pkl", "rb") as f:
    loaded_imputer = cloudpickle.load(f)

Y_test=np.ones(len(X_test))
X_test_clean,y_test_clean=loaded_imputer.transform(X_test,Y_test)
X_test_feature_selection=X_test_clean[selected_columns]
model = joblib.load('model.pkl')
y=model.predict(X_test_feature_selection)
