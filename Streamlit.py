import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('random_forest_model.pkl')

# Fonction pour faire des prédictions
def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# Interface utilisateur
st.title('Prédiction avec Random Forest')

# Nombre de caractéristiques d'entrée (à adapter selon vos données)
num_features = 5

# Exemple d'entrée utilisateur
input_data = []
for i in range(num_features):
    input_data.append(st.number_input(f'Feature {i+1}', value=0.0))

if st.button('Prédire'):
    prediction = predict(input_data)
    st.write(f'Prédiction: {prediction[0]}')
