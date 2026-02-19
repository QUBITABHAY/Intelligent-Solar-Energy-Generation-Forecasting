import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# create models directory if not there
models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# load dataset
data_path = '../data/spg.csv'
df = pd.read_csv(data_path)

df = df.drop_duplicates()

X = df.drop('generated_power_kw', axis=1)
y = df['generated_power_kw']

# split data and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# save model and scaler
joblib.dump(model, os.path.join(models_dir, 'random_forest_model.joblib'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))

print(f"Model and scaler saved successfully in '{models_dir}' directory.")
