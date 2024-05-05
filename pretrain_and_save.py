import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor
import pickle
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from joblib import Parallel, delayed
from sklearn.exceptions import NotFittedError

def create_and_optimize_model(model_name, model_class, param_dist):
    random_search = RandomizedSearchCV(model_class, param_distributions=param_dist, n_iter=6, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    random_search.fit(X_train_imputed, y_train)
    return (model_name, random_search.best_estimator_)

# Load the data
file_path = 'Delhi.csv'
processed_data = pd.read_csv(file_path)

# Define the target variable
target = "Price"
y_train = processed_data[target]

# Define the features including 'Location'
features = [
    "Area", "No. of Bedrooms", "Resale", "MaintenanceStaff", "Gymnasium",
    "SwimmingPool", "LandscapedGardens", "JoggingTrack", "RainWaterHarvesting",
    "IndoorGames", "ShoppingMall", "Intercom", "SportsFacility", "ATM",
    "ClubHouse", "School", "24X7Security", "PowerBackup", "CarParking",
    "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital", "WashingMachine",
    "Gasconnection", "AC", "Wifi", "Children'splayarea", "LiftAvailable", "BED",
    "VaastuCompliant", "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa",
    "Wardrobe", "Refrigerator", "Location"
]

# Extract the features
X_train = processed_data[features]

# One-hot encode the 'Location' feature
X_train = pd.get_dummies(X_train, columns=['Location'], drop_first=True)

# Define columns to impute
columns_to_impute = features[:-1]  # Exclude 'Location' as it's already encoded

# Replace 9 with NaN for imputation
for column in columns_to_impute:
    X_train.loc[X_train[column] == 9, column] = np.nan

# Imputation using SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")
X_train_imputed = imputer.fit_transform(X_train)

# Define parameter distributions for randomized search
# Define parameter distributions for randomized search
param_dists = {
    "Decision Tree": {},
    "Random Forest": {},
    "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    "Huber Regressor": {'epsilon': [1.0, 1.35], 'alpha': [0.0001, 0.001]},
    "CatBoost": {'depth': [6, 8], 'l2_leaf_reg': [1, 3, 5]}
}
# Create and optimize base models in parallel
base_models = Parallel(n_jobs=-1)(delayed(create_and_optimize_model)(model_name, model_class, param_dists[model_name])
                                  for model_name, model_class in [
                                      ("Decision Tree", DecisionTreeRegressor()),
                                      ("Random Forest", RandomForestRegressor()),
                                      ("KNN", KNeighborsRegressor()),
                                      ("Huber Regressor", HuberRegressor(max_iter=1000)),
                                      ("CatBoost", CatBoostRegressor(iterations=500, learning_rate=0.05, loss_function='MAE', verbose=0))
                                  ])

# Create a Voting Regressor with the optimized base models
voting_regressor = VotingRegressor(estimators=base_models, weights=[1, 7, 7, 1, 2])

models = {
    "Voting Regressor": voting_regressor
}

# Add optimized base models to the models dictionary
models.update({model_name: model for model_name, model in base_models})

# Define a location to save the models
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Fit and save each model using pickle
for model_name, model in models.items():
    try:
        model.fit(X_train_imputed, y_train)
        model_file_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"{model_name} trained, optimized, and saved successfully!")
    except NotFittedError as e:
        print(f"Error: {model_name} is not fitted yet. Call 'fit' with appropriate arguments before saving the model.")

print("All models trained, optimized, and saved successfully!")
