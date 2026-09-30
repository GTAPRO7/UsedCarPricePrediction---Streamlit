import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1. Load the data
try:
    df = pd.read_csv('CarDekhoCarDetails.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'CarDekhoCarDetails.csv' not found.")
    exit()

# 2. Select Features and Target
# We are predicting 'selling_price' based on name, year, km_driven, fuel, seller_type, transmission, owner
features = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
X = df[features].copy()
y = df['selling_price']

# 3. Encode Categorical Variables
name_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
seller_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
owner_encoder = LabelEncoder()

X['name'] = name_encoder.fit_transform(X['name'])
X['fuel'] = fuel_encoder.fit_transform(X['fuel'])
X['seller_type'] = seller_type_encoder.fit_transform(X['seller_type'])
X['transmission'] = transmission_encoder.fit_transform(X['transmission'])
X['owner'] = owner_encoder.fit_transform(X['owner'])

# 4. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model trained successfully with an R^2 score of: {score:.2f}")

# 5. Save the Model, Encoders, and car name list for deployment
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(name_encoder, 'name_encoder.pkl')
joblib.dump(fuel_encoder, 'fuel_encoder.pkl')
joblib.dump(seller_type_encoder, 'seller_type_encoder.pkl')
joblib.dump(transmission_encoder, 'transmission_encoder.pkl')
joblib.dump(owner_encoder, 'owner_encoder.pkl')

# Save sorted list of unique car names for the app dropdown
car_names = sorted(df['name'].unique().tolist())
joblib.dump(car_names, 'car_names.pkl')

print(f"Model and encoders saved to disk. ({len(car_names)} unique car names exported.)")