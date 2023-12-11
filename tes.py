import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
data = pd.read_csv("logika_fuzzy.csv")

# Extract relevant columns for input
features = data[['Seberapa sering Anda berkunjung ke Makassar untuk tujuan wisata kuliner?',
                  'Seberapa sering Anda mencari referensi tempat kuliner melalui internet atau dari rekomendasi teman dan keluarga?',
                  'Seberapa penting bagi Anda mendapatkan rekomendasi tempat kuliner yang sesuai dengan preferensi makanan yang Anda sukai?']]

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Load the pre-trained model
model = joblib.load('kmeans_model.pkl')

# UI
st.title('Restaurant Recommendation App')

# Create a form for user input
st.subheader('User Preferences')
visit_frequency = st.radio("Seberapa sering Anda berkunjung ke Makassar untuk tujuan wisata kuliner?",
                          ["Tidak Sering/Tidak Penting", "Jarang", "Netral", "Sering/Penting", "Sangat Sering/Sangat Penting"])

search_frequency = st.radio("Seberapa sering Anda mencari referensi tempat kuliner melalui internet atau dari rekomendasi teman dan keluarga?",
                            ["Tidak Sering/Tidak Penting", "Jarang", "Netral", "Sering/Penting", "Sangat Sering/Sangat Penting"])

food_preference = st.radio("Seberapa penting bagi Anda mendapatkan rekomendasi tempat kuliner yang sesuai dengan preferensi makanan yang Anda sukai?",
                           ["Tidak Sering/Tidak Penting", "Jarang", "Netral", "Sering/Penting", "Sangat Sering/Sangat Penting"])

# Convert user input to numeric values
map_change = {'Tidak Sering/Tidak Penting': 1, 'Jarang': 2, 'Netral': 3, 'Sering/Penting': 4, 'Sangat Sering/Sangat Penting': 5}
user_input = pd.Series([map_change[visit_frequency], map_change[search_frequency], map_change[food_preference]])

# Apply the pre-trained model to user input
user_input_scaled = scaler.transform(user_input.values.reshape(1, -1))
predicted_class = model.predict(user_input_scaled)[0]

# Display app result based on the model's prediction
if predicted_class == 1:
    st.success("Buat aplikasi karena hasil prediksi model adalah 1")
else:
    st.warning("Tidak perlu membuat aplikasi karena hasil prediksi model adalah 0")