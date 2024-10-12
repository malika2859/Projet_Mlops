import streamlit as st
import pickle
import pandas as pd
import os

# Charger le modèle depuis le fichier
model_path = os.path.join(os.getcwd(), "random_forest_model.pkl")  # Chemin relatif au répertoire courant

try:
    model = pickle.load(open(model_path, "rb"))
    st.success("Model loaded successfully from {}".format(model_path))
except FileNotFoundError:
    st.error("File not found: {}".format(model_path))
    st.stop()

# Fonction pour effectuer la prédiction
def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])

# Interface utilisateur
st.title("Prédiction avec le modèle Random Forest")
st.write("Entrez les caractéristiques pour obtenir une prédiction.")

# Formulaire pour saisir les caractéristiques
credit_lines_outstanding = st.number_input("Lignes de crédit en cours :", min_value=0)
loan_amt_outstanding = st.number_input("Montant du prêt en cours :", min_value=0.0, step=0.01)
total_debt_outstanding = st.number_input("Total de la dette en cours :", min_value=0.0, step=0.01)
income = st.number_input("Revenu :", min_value=0.0, step=0.01)
years_employed = st.number_input("Années d'emploi :", min_value=0)
fico_score = st.number_input("Score FICO :", min_value=0)

# Bouton pour effectuer la prédiction
if st.button("Prédire"):
    features = [credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]
    prediction = model_pred(features)

    if prediction == 1:
        st.error("Le client pourrait être en défaut de paiement.")
    else:
        st.success("Le client ne présente pas de risque de défaut.")
