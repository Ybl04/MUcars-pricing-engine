import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="MUCars Pricing Engine", page_icon="🚗", layout="wide")

API_URL = "http://127.0.0.1:8000"  # will change to the live Render URL once deployed

st.title("🚗 MUCars — Moteur d'Intelligence Tarifaire")
st.caption("Estimation de prix pour le marché marocain des voitures d'occasion, basée sur 75 000+ annonces réelles (MUCars-2024)")

BRANDS = ['Alfa Romeo', 'Audi', 'BMW', 'BYD', 'Bentley', 'Cadillac', 'Chery', 'Chevrolet',
          'Chrysler', 'Citroen', 'Cupra', 'DS', 'Dacia', 'Daewoo', 'Daihatsu', 'Dodge', 'Fiat',
          'Ford', 'GMC', 'Geely', 'Honda', 'Hummer', 'Hyundai', 'Infiniti', 'Isuzu', 'Iveco',
          'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes-Benz',
          'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Seat',
          'Skoda', 'Smart', 'Ssangyong', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen',
          'Volvo', 'mini', 'Autres']

MODELS = ['19', '190', '206', '208', '220', '308', '500', 'Accent', 'Astra', 'Berlingo', 'C3',
          'Caddy', 'Classe C', 'Clio', 'Dokker', 'Duster', 'Fiesta', 'Focus', 'Golf 4', 'Golf 7',
          'Kangoo', 'Kuga', 'Logan', 'Megane', 'Octavia', 'Palio', 'Partner', 'Passat', 'Picanto',
          'Polo', 'Punto', 'Qashqai', 'Range Rover Evoque', 'Sandero', 'Sandero Stepway',
          'Santa Fe', 'Tiguan', 'Touareg', 'Tucson', 'Uno', 'Other']

LOCATIONS = ['Agadir', 'Berrechid', 'Béni Mellal', 'Casablanca', 'El Jadida', 'Fès', 'Khouribga',
             'Kénitra', 'Laâyoune', 'Marrakech', 'Meknès', 'Mohammedia', 'Nador', 'Oujda', 'Rabat',
             'Safi', 'Salé', 'Tanger', 'Temara', 'Tétouan', 'Other']

with st.sidebar:
    st.header("Caractéristiques du véhicule")

    brand = st.selectbox("Marque", BRANDS, index=BRANDS.index('Dacia'))
    model_grouped = st.selectbox("Modèle", MODELS, index=MODELS.index('Logan'))
    year = st.slider("Année", 1995, 2024, 2019)
    mileage = st.number_input("Kilométrage (km)", min_value=0, max_value=550000, value=85000, step=5000)
    fiscal_power = st.number_input("Puissance fiscale (CV)", min_value=4, max_value=45, value=6)

    condition = st.selectbox("État", ['Excellent', 'Very Good', 'Good', 'Fair', 'New', 'Damaged', 'For Parts', 'Unknown'])
    gearbox = st.radio("Boîte de vitesses", ['Manual', 'Automatic'])
    fuel = st.selectbox("Carburant", ['Diesel', 'Petrol', 'Hybrid', 'Electrique', 'LPG'])
    origin = st.selectbox("Origine", ['WW in Morocco', 'Customs-cleared car', 'Imported New', 'Car not yet customs-cleared', 'Unknown'])
    first_owner = st.radio("Premier propriétaire", ['Yes', 'No', 'Unknown'])
    doors = st.radio("Nombre de portes", [5.0, 3.0])
    location = st.selectbox("Ville", LOCATIONS)

    st.divider()
    st.subheader("Équipements")
    equip_cols = st.columns(2)
    equipment_options = ["ABS", "Air_Conditioning", "Airbags", "Alloy_Wheels", "CD_MP3_Bluetooth",
                          "Central_Locking", "Cruise_Control", "ESP", "Electric_Windows",
                          "Leather_Seats", "Navigation_System_GPS", "Onboard_Computer",
                          "Parking_Sensors", "Rear_Camera", "Speed_Limiter", "Sunroof"]
    has_equipment = {}
    for i, eq in enumerate(equipment_options):
        with equip_cols[i % 2]:
            has_equipment[eq] = st.checkbox(eq.replace("_", " "), value=False)

    submitted = st.button("Estimer le prix", type="primary", use_container_width=True)

# ---- Main panel ----
if submitted:
    # Build the request payload from sidebar inputs
    payload = {
        "Brand": brand,
        "Model_grouped": model_grouped,
        "Year": year,
        "Condition": condition,
        "Gearbox": gearbox,
        "Fuel": fuel,
        "Number_of_Doors": doors,
        "Origin": origin,
        "First_Owner": first_owner,
        "Mileage_numeric": float(mileage),
        "Fiscal_Power_numeric": float(fiscal_power),
        "Location_grouped": location,
        **{f"has_{eq}": int(v) for eq, v in has_equipment.items()}
    }

    with st.spinner("Analyse en cours..."):
        try:
            pred_resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            expl_resp = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
            comp_resp = requests.get(f"{API_URL}/compare", params={
                "brand": brand,
                "model_grouped": model_grouped,
                "year": year,
                "mileage": float(mileage)
            }, timeout=30)

            pred = pred_resp.json()
            expl = expl_resp.json()
            comp = comp_resp.json()

        except requests.exceptions.ConnectionError:
            st.error("Impossible de se connecter à l'API. Assurez-vous que le serveur FastAPI est en cours d'exécution sur http://127.0.0.1:8000")
            st.stop()
        except Exception as e:
            st.error(f"Erreur inattendue: {e}")
            st.stop()

    # ---- Row 1: Price estimate ----
    st.subheader("Estimation du prix")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Prix estimé",
            value=f"{pred['predicted_price_mad']:,} MAD"
        )
    with col2:
        st.metric(
            label="Fourchette basse",
            value=f"{pred['confidence_interval_mad'][0]:,} MAD"
        )
    with col3:
        st.metric(
            label="Fourchette haute",
            value=f"{pred['confidence_interval_mad'][1]:,} MAD"
        )

    st.divider()

    # ---- Row 2: SHAP explanation + similar listings side by side ----
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Facteurs influençant le prix")
        st.caption("Contribution de chaque caractéristique à l'estimation (en MAD)")

        shap_data = expl['shap_contributions_mad']

        # Keep top 10 by absolute value, skip near-zero noise
        FEATURE_LABELS = {
    'Year': 'Année',
    'Brand': 'Marque',
    'Model_grouped': 'Modèle',
    'Gearbox': 'Boîte de vitesses',
    'Fuel': 'Carburant',
    'Fiscal_Power_numeric': 'Puissance fiscale',
    'Condition': 'État',
    'Mileage_numeric': 'Kilométrage',
    'mileage_per_year': 'Km/an',
    'car_age': 'Âge du véhicule',
    'Origin': 'Origine',
    'First Owner': 'Premier propriétaire',
    'Location_grouped': 'Ville',
    'Number of Doors': 'Nb. portes',
    }

        top_features = dict(
            sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

        top_features_labeled = {
            FEATURE_LABELS.get(k, k): v
            for k, v in top_features.items()
        }

        features = list(top_features_labeled.keys())
        values = list(top_features_labeled.values())
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+,} MAD" for v in values],
            textposition='auto'
        ))

        fig.update_layout(
            xaxis_title="Contribution (MAD)",
            yaxis=dict(autorange='reversed'),
            margin=dict(l=20, r=80, t=20, b=20),
            height=380,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("Annonces similaires sur le marché")
        st.caption(f"Véhicules {brand} similaires en année et kilométrage")

        if comp:
            comp_df = pd.DataFrame(comp)
            comp_df['Price'] = comp_df['Price'].apply(lambda x: f"{int(x):,} MAD")
            comp_df['Mileage_numeric'] = comp_df['Mileage_numeric'].apply(lambda x: f"{int(x):,} km")
            comp_df = comp_df.rename(columns={
                'Brand': 'Marque',
                'Model_grouped': 'Modèle',
                'Year': 'Année',
                'Price': 'Prix',
                'Mileage_numeric': 'Kilométrage',
                'Location_grouped': 'Ville'
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune annonce similaire trouvée pour ces critères.")

    st.divider()

    # ---- Row 3: Summary interpretation ----
    st.subheader("Interprétation")
    top_pos = [(k, v) for k, v in shap_data.items() if v > 0]
    top_neg = [(k, v) for k, v in shap_data.items() if v < 0]
    top_pos_sorted = sorted(top_pos, key=lambda x: x[1], reverse=True)[:2]
    top_neg_sorted = sorted(top_neg, key=lambda x: x[1])[:2]

    pos_text = " et ".join([f"**{k}** (+{v:,} MAD)" for k, v in top_pos_sorted])
    neg_text = " et ".join([f"**{k}** ({v:,} MAD)" for k, v in top_neg_sorted])

    st.markdown(
        f"Le prix estimé de **{pred['predicted_price_mad']:,} MAD** est principalement "
        f"porté à la hausse par {pos_text}, "
        f"et tiré à la baisse par {neg_text}."
    )

else:
    # Default state — shown before the user clicks Estimate
    st.info("👈 Renseignez les caractéristiques du véhicule dans le panneau de gauche, puis cliquez sur **Estimer le prix**.")
    st.markdown("""
    #### Comment ça fonctionne ?
    1. **Sélectionnez** les caractéristiques de votre véhicule dans le panneau latéral
    2. **Cliquez** sur *Estimer le prix* pour obtenir une estimation basée sur 75 000+ annonces réelles
    3. **Explorez** les facteurs qui influencent le prix grâce à l'analyse SHAP
    4. **Comparez** avec des annonces similaires sur le marché marocain
    """)