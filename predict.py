import joblib
import pandas as pd

# Load saved model and encoders
model = joblib.load('car_price_model.pkl')
name_encoder = joblib.load('name_encoder.pkl')
fuel_encoder = joblib.load('fuel_encoder.pkl')
seller_type_encoder = joblib.load('seller_type_encoder.pkl')
transmission_encoder = joblib.load('transmission_encoder.pkl')
owner_encoder = joblib.load('owner_encoder.pkl')

def get_prediction(car_name, year, km_driven, fuel, seller_type, transmission, owner):
    """Encodes categorical inputs and returns the predicted selling price."""
    
    # Transform text inputs using the fitted encoders
    name_encoded = name_encoder.transform([car_name])[0]
    fuel_encoded = fuel_encoder.transform([fuel])[0]
    seller_type_encoded = seller_type_encoder.transform([seller_type])[0]
    transmission_encoded = transmission_encoder.transform([transmission])[0]
    owner_encoded = owner_encoder.transform([owner])[0]
    
    input_data = pd.DataFrame({
        'name': [name_encoded],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel_encoded],
        'seller_type': [seller_type_encoded],
        'transmission': [transmission_encoded],
        'owner': [owner_encoded]
    })
    
    prediction = model.predict(input_data)
    return prediction[0]