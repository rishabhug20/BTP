import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request, url_for, redirect
import pickle
import googlemaps
from sklearn.exceptions import NotFittedError

# Load Google Maps API key from environment variable
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))
# Function to read data from CSV
def wrangle(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

# Load the data
file_path = 'Delhi.csv'
processed_data = wrangle(file_path)

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

# Initialize Flask app
app = Flask(__name__)

# Initialize empty dictionary to hold models
models = {}

# Load models from pickled files
models_dir = 'models'
for model_file in os.listdir(models_dir):
    if model_file.endswith('.pkl'):
        model_name = model_file.split('.')[0]
        with open(os.path.join(models_dir, model_file), 'rb') as file:
            models[model_name] = pickle.load(file)

# Function to make prediction using loaded models
def make_prediction(data):
    data = pd.get_dummies(data, columns=['Location'], drop_first=True)
    data = data.reindex(columns=X_train.columns, fill_value=0)
    data_imputed = imputer.transform(data)
    
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(data_imputed)[0]
        predictions[model_name] = round(prediction, 2)
    
    # Ensembled prediction
    ensembled_prediction = np.mean(list(predictions.values()))
    predictions['Ensembled'] = round(ensembled_prediction, 2)
    
    return predictions

# Route to render UI template with the form
@app.route('/')
def home():
    banner_image_url = url_for('static', filename='GOV_banner.png')
    favicon_ico = url_for('static', filename='favicon.ico')
    return render_template('home.html', banner_image_url=banner_image_url, favicon_ico=favicon_ico)

# Route to render input form template
@app.route('/inputs')
def inputs():
    locations = processed_data['Location'].unique().tolist()
    favicon_ico = url_for('static', filename='favicon.ico')
    return render_template('inputs.html', locations=locations, favicon_ico=favicon_ico)

# Route to render about us template
@app.route('/aboutus')
def about():
    team_members = {
        'Jatin Jangid': {
            'image_url': url_for('static', filename='profile-pic4.png'),
            'email': 'jatin.jangid.ug20@nsut.ac.in'
        },
        'Shreyansh Tiwari': {
            'image_url': url_for('static', filename='profile-pic3.png'),
            'email': 'shreyansh.tiwari.ug20@nsut.ac.in'
        },
        'Rishabh Jain': {
            'image_url': url_for('static', filename='profile-pic2.png'),
            'email': 'rishabh.jain.ug20@nsut.ac.in'
        },
        'Chandra Pratap': {
            'image_url': url_for('static', filename='profile-pic1.png'),
            'email': 'chandra.pratap.ug20@nsut.ac.in'
        },
        'bg-image': url_for('static', filename='10974.jpg')
    }

    favicon_ico = url_for('static', filename='favicon.ico')

    return render_template('aboutus.html', team_members=team_members, favicon_ico=favicon_ico)

# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Area": float(request.form.get('area', 0)),
        "No. of Bedrooms": int(request.form.get('bedrooms', 0)),
        "Resale": int(request.form.get('Resale', 0)),
        "MaintenanceStaff": int(request.form.get('maintenance', 0)),
        "Gymnasium": int(request.form.get('gymnasium', 0)),
        "SwimmingPool": int(request.form.get('SwimmingPool', 0)),
        "LandscapedGardens": int(request.form.get('LandscapedGardens', 0)),
        "JoggingTrack": int(request.form.get('jogging_track', 0)),
        "RainWaterHarvesting": int(request.form.get('rainwater_harvesting', 0)),
        "IndoorGames": int(request.form.get('indoor_games', 0)),
        "ShoppingMall": int(request.form.get('shopping_mall', 0)),
        "Intercom": int(request.form.get('intercom', 0)),
        "SportsFacility": int(request.form.get('sports_facility', 0)),
        "ATM": int(request.form.get('atm', 0)),
        "ClubHouse": int(request.form.get('club_house', 0)),
        "School": int(request.form.get('school', 0)),
        "24X7Security": int(request.form.get('security', 0)),
        "PowerBackup": int(request.form.get('power_backup', 0)),
        "CarParking": int(request.form.get('car_parking', 0)),
        "StaffQuarter": int(request.form.get('staff_quarter', 0)),
        "Cafeteria": int(request.form.get('cafeteria', 0)),
        "MultipurposeRoom": int(request.form.get('multipurpose_room', 0)),
        "Hospital": int(request.form.get('hospital', 0)),
        "WashingMachine": int(request.form.get('washing_machine', 0)),
        "Gasconnection": int(request.form.get('gas_connection', 0)),
        "AC": int(request.form.get('ac', 0)),
        "Wifi": int(request.form.get('wifi', 0)),
        "Children'splayarea": int(request.form.get('play_area', 0)),
        "LiftAvailable":int(request.form.get('lift_available', 0)),
        "BED": int(request.form.get('bed', 0)),
        "VaastuCompliant": int(request.form.get('vaastu_compliant', 0)),
        "Microwave": int(request.form.get('microwave', 0)),
        "GolfCourse": int(request.form.get('golf_course', 0)),
        "TV": int(request.form.get('tv', 0)),
        "DiningTable": int(request.form.get('dining_table', 0)),
        "Sofa": int(request.form.get('sofa', 0)),
        "Wardrobe": int(request.form.get('wardrobe', 0)),
        "Refrigerator": int(request.form.get('refrigerator', 0)),
        "Location": request.form.get('location', '')
    }

    df = pd.DataFrame(data, index=[0])
    prediction = make_prediction(df)

    # Calculate average predicted price
    avg_predicted_price = sum(prediction.values()) / len(prediction)

    # Remove 'Ensembled' key from prediction
    prediction.pop('Ensembled', None)

    # Geocode the predicted location to get its coordinates
    geocode_result = gmaps.geocode(data['Location'])

    if geocode_result:
        location_details = geocode_result[0]
        latitude = location_details['geometry']['location']['lat']
        longitude = location_details['geometry']['location']['lng']

        # Render result.html with predictions and maps_html_content
        favicon_ico = url_for('static', filename='favicon.ico')
        bg_image_ = url_for('static', filename='bg.jpg')
        return render_template('result.html', favicon_ico=favicon_ico, latitude=latitude, longitude=longitude,
                               location=data['Location'], predictions=prediction, avg_predicted_price=round(avg_predicted_price,2),
                               key=os.getenv('GOOGLE_MAPS_API_KEY'),
                               bg_image_=bg_image_)
    else:
        return redirect('/inputs')

if __name__ == '__main__':
    app.run(debug=False)
