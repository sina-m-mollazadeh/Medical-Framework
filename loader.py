# Date Column Section
import pandas as pd
import jdatetime #Convert From Jalali to Datetime

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

def process_date_columns(data, reference_column=None):
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




# Conversion Section
import re

def ConvertToNumeric(data,y_column):
    data = process_date_columns(data)
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    categorical_columns = list(set(cols) - set(num_cols))

    all_mappings = {}
    y_mappings={}
    for column in categorical_columns:
        categories = list(data[column].dropna().astype(str).unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        all_mappings[column] = mapping
        if(column==y_column):
            y_mappings[column]=mapping
        

        data[column] = data[column].map(lambda x: mapping.get(str(x), x))

    return data, all_mappings,y_mappings

def sanitize_column_names(data, y_column):
    data.columns = [
        col if col == y_column else re.sub(r'[^\w]', '_', col)
        for col in data.columns
    ]
    return data

import numpy as np
import pandas as pd
from collections import Counter

def get_adaptive_weights(X, Y):
    # Identify the minority class ratio
    counts = Y.value_counts(normalize=True)
    minority_ratio = counts.min()

    # If the data is highly imbalanced, Recall is our priority.
    # We set a static, high-quality objective.
    if minority_ratio < 0.20:
        # Heavily prioritize Recall and F-Beta
        return {
            'acc': 0.1, 
            'rec': 0.5, 
            'prec': 0.1, 
            'f1': 0.3
        }
    else:
        # Balanced data
        return {'acc': 0.25, 'rec': 0.25, 'prec': 0.25, 'f1': 0.25}

from sklearn.preprocessing import LabelEncoder # Encode y column
def load_data(path, y_column):
    if("csv" in path):
        data=pd.read_csv(path, na_values=[" ", "", "NA", "NaN"])
    elif("xlsx" in path):
        data=pd.read_excel(path,na_values=[" ", "", "NA", "NaN"])
    else:
        print("Format not Supported")
        return None
    
    # The below three lines remove columns with more than Threshold% Empty Rows
    threshold = 0.9
    valid_cols = data.columns[data.notna().mean() >= threshold]
    data = data[valid_cols]

    # Drop Column whose y values have null
    data = data.dropna(subset=[y_column])    

    data=sanitize_column_names(data,y_column)

    data,all_mappings,y_mappings=ConvertToNumeric(data,y_column=y_column)
    le = LabelEncoder()

    Y=data[y_column]
    Y = le.fit_transform(Y)
    X=data.drop(columns=y_column)

    weights=get_adaptive_weights(X,pd.Series(Y,name="a"))

    return X,Y,data,all_mappings,y_mappings,weights