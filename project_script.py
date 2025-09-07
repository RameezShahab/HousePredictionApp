# Auto-extracted from Project.ipynb
# You may need to edit this file to remove notebook magics (e.g. %matplotlib inline)

# --- cell 1 ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# --- cell 2 ---
df = pd.read_csv('AmesHousing.csv')
df.head()

# --- cell 3 ---
df.info()

# --- cell 4 ---
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']


# --- cell 5 ---
train_df, test_df, train_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)


# --- cell 6 ---
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

# --- cell 7 ---
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])



# --- cell 8 ---
# from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)


# --- cell 9 ---
model1 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# --- cell 10 ---
model2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# --- cell 11 ---
model3 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=42, enable_categorical=False))
])


# --- cell 12 ---
models = {
    "Linear Regression": model1,
    "Random Forest": model2,
    "XGBoost": model3
}

results = []

# --- cell 13 ---
for name, model in models.items():
    # Fit the model
    model.fit(train_df, train_target)
    
    # Predict on test set
    preds = model.predict(test_df)
        # Evaluate the model
     
    rmse = np.sqrt(mean_squared_error(test_target, preds))
    r2 = r2_score(test_target, preds)
    
    results.append((name, rmse, r2))

print(f"{'Model':<20} {'RMSE':<15} {'R² Score':<15}")
for name, rmse, r2 in results:
    print(f"{name:<20} {rmse:<15.4f} {r2:<15.4f}")

# --- cell 14 ---


# --- cell 15 ---
# Add this at the end of project_script.py to make variables accessible
if __name__ == "__main__":
    # Make these variables global for import
    global train_df, test_df, train_target, test_target, models
    print("Script loaded successfully!")
