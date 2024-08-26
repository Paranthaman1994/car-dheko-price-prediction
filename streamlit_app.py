import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image

# Load the dataset
df = pd.read_csv('car_dheko_final.csv')
df.drop(columns=["Unnamed: 0"], inplace=True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stSelectbox, .stNumberInput {
        background-color: #f0f0f0;
        color: #333333;
        font-size: 16px;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #FF6F6F;
    }
    </style>
""", unsafe_allow_html=True)

# # Sidebar Logo and Banner
# st.sidebar.image("carDekho-newLogo.svg", use_column_width=True)
# banner = Image.open('banner.jpeg')
# banner = banner.resize((2500, 900))
# st.image(banner)

# # Sidebar Description
# st.sidebar.header("About Cardheko")
# st.sidebar.write("""
# Cardheko is a comprehensive platform that helps users explore, compare, and purchase cars.
# With an extensive range of new and used cars, the website offers in-depth reviews, 
# pricing details, and expert advice to guide car buyers in making informed decisions. 
# Use our price prediction tool to estimate car values based on various features and conditions.
# """)

# Streamlit app interface
st.title("Car Price Prediction")

st.header("Enter Car Details")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    bt = st.selectbox('Body Type (bt)', df['bt'].unique(), key='bt')
    km = st.number_input('Kilometers Driven (km)', min_value=0, max_value=500000, step=1000, key='km')
    owner = st.number_input('Owner Number', min_value=1, max_value=8, key='owner')
    oem = st.selectbox('OEM', df['oem'].unique(), key='oem')
    model = st.selectbox('Model', df['model'].unique(), key='model')
    modelyear = st.number_input('Model Year', min_value=int(df['modelyear'].min()), max_value=int(df['modelyear'].max()), key='modelyear')
    state = st.selectbox('State', df['state'].unique(), key='state')

with col2:
    insurance_validity = st.selectbox('Insurance Validity', df['insurance_validity'].unique(), key='insurance_validity')
    fuel_type = st.selectbox('Fuel Type', df['fuel_type'].unique(), key='fuel_type')
    seats = st.number_input('Seats', min_value=2, max_value=10, step=1, key='seats')
    rto = st.selectbox('RTO', df['rto'].unique(), key='rto')
    engine_displacement = st.number_input('Engine Displacement (cc)', min_value=500, max_value=5000, step=10, key='engine_displacement')
    mileage = st.number_input('Mileage (kmpl)', min_value=5.0, max_value=50.0, step=0.1, key='mileage')

# Label Encoding for Categorical Variables
categorical_cols = df.select_dtypes(include=[object]).columns
encoders = {}
for column in categorical_cols:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Splitting the data into features and target
X = df.drop(columns='price')
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Training the model
model = GradientBoostingRegressor(n_estimators=300, random_state=42, learning_rate=0.2, max_depth=5, min_samples_leaf=4, min_samples_split=2)
model.fit(X_train, y_train)

# Creating a dictionary for the input data
input_data = {
    'bt': bt,
    'km': km,
    'owner': owner,
    'oem': oem,
    'model': model,
    'modelyear': modelyear,
    'state': state,
    'insurance_validity': insurance_validity,
    'fuel_type': fuel_type,
    'seats': seats,
    'rto': rto,
    'engine_displacement': engine_displacement,
    'mileage': mileage
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Apply label encoding to the input data using the fitted encoders
for column in input_df.columns:
    if column in encoders:
        le = encoders[column]
        if not input_df[column].isin(le.classes_).all():
            # Handle unseen labels - map them to a default or unknown category
            input_df[column] = input_df[column].apply(lambda x: le.classes_[0] if x not in le.classes_ else x)
        input_df[column] = le.transform(input_df[column])

# Make prediction using the trained model
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Price: ₹{int(prediction[0]):,}")


