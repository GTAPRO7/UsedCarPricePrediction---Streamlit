import streamlit as st
import joblib
from predict import get_prediction

st.set_page_config(page_title="Used Car Valuation", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.write("Enter the vehicle specifications below to estimate its resale market value based on the CarDekho dataset.")

# Load car names from the dataset for the dropdown
car_names = joblib.load('car_names.pkl')

# Input Form mapped to Kaggle Dataset Features
with st.form("prediction_form"):
    # Car selection at the top — searchable dropdown with all cars from the dataset
    car_name = st.selectbox("Select Car Model", options=car_names, index=0)

    col1, col2 = st.columns(2)
    
    with col1:
        year = st.slider("Year of Manufacture", min_value=2000, max_value=2024, value=2015)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=40000, step=500)
        fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
        
    with col2:
        seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
        transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
        owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    
    submit_button = st.form_submit_button(label="Predict Selling Price")

# Output Processing
if submit_button:
    try:
        predicted_price = get_prediction(car_name, year, km_driven, fuel, seller_type, transmission, owner)
        
        # Ensure prediction is not negative
        predicted_price = max(predicted_price, 0)
        
        st.success("Prediction Complete!")
        # Display price in Lakhs (1 Lakh = 100,000 INR) — standard Indian car pricing format
        price_in_lakhs = predicted_price / 100000
        st.metric(label=f"Estimated Resale Value for {car_name}", value=f"₹ {price_in_lakhs:.2f} Lakhs")
        st.caption(f"(₹ {predicted_price:,.0f})")
        
        st.write("---")
        st.caption("This valuation engine is powered by a Random Forest algorithm trained on real Indian automotive market data from CarDekho.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")