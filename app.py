"""
Application Streamlit pour prédire l'attrition bancaire en temps réel
Utilise le modèle Spark MLlib sauvegardé

Pour lancer l'application :
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.functions import col

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================

st.set_page_config(
    page_title="Prédiction Attrition Bancaire",
    page_icon="🏦",
    layout="wide"
)

# ============================================
# INITIALISATION DE SPARK ET DU MODÈLE
# ============================================

@st.cache_resource
def init_spark():
    """Initialise la session Spark (une seule fois)"""
    spark = SparkSession.builder \
        .appName("Attrition Prediction App") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    return spark

@st.cache_resource
def load_model(_spark):
    """Charge le modèle sauvegardé (une seule fois)"""
    try:
        model = CrossValidatorModel.load("models/best_model_attrition")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Initialiser Spark et charger le modèle
spark = init_spark()
model = load_model(spark)

# ============================================
# INTERFACE UTILISATEUR
# ============================================

st.title("🏦 Prédiction de l'Attrition Bancaire")
st.markdown("---")

# Vérifier que le modèle est chargé
if model is None:
    st.error("❌ Impossible de charger le modèle. Vérifiez le chemin : 'models/best_model_attrition'")
    st.stop()

st.success("✅ Modèle chargé avec succès!")

# ============================================
# MODE DE PRÉDICTION
# ============================================

mode = st.radio(
    "Choisissez le mode de prédiction :",
    ["🧑 Prédiction individuelle", "📊 Prédiction par lot (CSV)"],
    horizontal=True
)

# ============================================
# MODE 1 : PRÉDICTION INDIVIDUELLE
# ============================================

if mode == "🧑 Prédiction individuelle":
    st.subheader("Entrez les informations du client :")
    
    # Créer 3 colonnes pour une meilleure présentation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📋 Informations de base**")
        credit_score = st.number_input(
            "Score de crédit", 
            min_value=300, 
            max_value=850, 
            value=650,
            help="Score de crédit du client (300-850)"
        )
        age = st.number_input(
            "Âge", 
            min_value=18, 
            max_value=100, 
            value=35,
            help="Âge du client"
        )
        tenure = st.number_input(
            "Ancienneté (années)", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="Nombre d'années avec la banque"
        )
        balance = st.number_input(
            "Solde du compte (€)", 
            min_value=0.0, 
            value=50000.0,
            help="Solde actuel du compte"
        )
    
    with col2:
        st.markdown("**💳 Produits & Services**")
        num_products = st.selectbox(
            "Nombre de produits", 
            [1, 2, 3, 4],
            help="Nombre de produits bancaires détenus"
        )
        has_card = st.selectbox(
            "Possède une carte de crédit ?", 
            ["Oui", "Non"]
        )
        is_active = st.selectbox(
            "Membre actif ?", 
            ["Oui", "Non"]
        )
        estimated_salary = st.number_input(
            "Salaire estimé (€)", 
            min_value=0.0, 
            value=100000.0,
            help="Salaire annuel estimé"
        )
    
    with col3:
        st.markdown("**🌍 Profil**")
        gender = st.selectbox(
            "Genre", 
            ["Homme", "Femme"]
        )
        geography = st.selectbox(
            "Pays", 
            ["Espagne", "France", "Allemagne"]
        )
    
    # Bouton de prédiction
    st.markdown("---")
    
    if st.button("🔮 Prédire le risque d'attrition", type="primary", use_container_width=True):
        
        # Préparer les données
        # Convertir les valeurs en format attendu par le modèle
        has_card_val = 1.0 if has_card == "Oui" else 0.0
        is_active_val = 1.0 if is_active == "Oui" else 0.0
        gender_index = 1.0 if gender == "Femme" else 0.0
        geo_france = 1.0 if geography == "France" else 0.0
        geo_germany = 1.0 if geography == "Allemagne" else 0.0
        
        # Créer un DataFrame Pandas
        input_data = pd.DataFrame({
            'CreditScore': [float(credit_score)],
            'Age': [float(age)],
            'Tenure': [float(tenure)],
            'Balance': [float(balance)],
            'NumOfProducts': [float(num_products)],
            'HasCrCard': [has_card_val],
            'IsActiveMember': [is_active_val],
            'EstimatedSalary': [float(estimated_salary)],
            'GenderIndex': [gender_index],
            'Geography_France': [geo_france],
            'Geography_Germany': [geo_germany]
        })
        
        # Convertir en DataFrame Spark
        spark_df = spark.createDataFrame(input_data)
        
        # Ajouter une colonne label factice (requise par le pipeline)
        spark_df = spark_df.withColumn("label", col("CreditScore") * 0)
        
        # Faire la prédiction
        with st.spinner("🔄 Prédiction en cours..."):
            prediction_df = model.transform(spark_df)
            
            # Récupérer les résultats
            result = prediction_df.select("prediction", "probability").collect()[0]
            prediction = int(result["prediction"])
            probability = result["probability"].toArray()
            
            # Probabilité de départ (classe 1)
            prob_churn = probability[1] * 100
            prob_stay = probability[0] * 100
        
        # Afficher les résultats
        st.markdown("---")
        st.subheader("📊 Résultats de la prédiction")
        
        # Créer 2 colonnes pour les résultats
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error("⚠️ **RISQUE ÉLEVÉ D'ATTRITION**")
                st.markdown(f"**Probabilité de départ : {prob_churn:.2f}%**")
            else:
                st.success("✅ **FAIBLE RISQUE D'ATTRITION**")
                st.markdown(f"**Probabilité de rester : {prob_stay:.2f}%**")
        
        with res_col2:
            st.metric(
                label="Score de risque", 
                value=f"{prob_churn:.1f}%",
                delta=f"{prob_churn - 50:.1f}% vs moyenne",
                delta_color="inverse"
            )
        
        # Barre de progression
        st.progress(prob_churn / 100)
        
        # Recommandations
        st.markdown("---")
        st.subheader("💡 Recommandations")
        
        if prediction == 1:
            st.warning("""
            **Actions recommandées pour réduire le risque d'attrition :**
            - 📞 Contacter le client pour comprendre ses besoins
            - 🎁 Proposer des offres personnalisées ou promotions
            - 💬 Améliorer la relation client (satisfaction, support)
            - 📊 Analyser l'utilisation des produits actuels
            """)
        else:
            st.info("""
            **Client à faible risque :**
            - ✅ Maintenir la qualité du service
            - 🔄 Proposer des produits complémentaires adaptés
            - 📈 Surveiller régulièrement la satisfaction
            """)

# ============================================
# MODE 2 : PRÉDICTION PAR LOT (CSV)
# ============================================

elif mode == "📊 Prédiction par lot (CSV)":
    st.subheader("Téléchargez un fichier CSV contenant les données clients")
    
    # Afficher le format attendu
    with st.expander("📋 Format du fichier CSV attendu"):
        st.markdown("""
        Le fichier CSV doit contenir les colonnes suivantes :
        - `CreditScore` : Score de crédit (nombre)
        - `Age` : Âge (nombre)
        - `Tenure` : Ancienneté en années (nombre)
        - `Balance` : Solde du compte (nombre)
        - `NumOfProducts` : Nombre de produits (1-4)
        - `HasCrCard` : Possède carte de crédit (0 ou 1)
        - `IsActiveMember` : Membre actif (0 ou 1)
        - `EstimatedSalary` : Salaire estimé (nombre)
        - `GenderIndex` : Genre (0=Homme, 1=Femme)
        - `Geography_France` : En France (0 ou 1)
        - `Geography_Germany` : En Allemagne (0 ou 1)
        """)
        
        # Exemple de données
        st.markdown("**Exemple de données :**")
        example_df = pd.DataFrame({
            'CreditScore': [650, 720],
            'Age': [35, 42],
            'Tenure': [5, 7],
            'Balance': [50000, 0],
            'NumOfProducts': [2, 1],
            'HasCrCard': [1, 1],
            'IsActiveMember': [1, 0],
            'EstimatedSalary': [100000, 80000],
            'GenderIndex': [0, 1],
            'Geography_France': [0, 1],
            'Geography_Germany': [0, 0]
        })
        st.dataframe(example_df)
    
    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV", 
        type="csv",
        help="Le fichier doit contenir toutes les colonnes requises"
    )
    
    if uploaded_file is not None:
        try:
            # Lire le CSV
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Fichier chargé : {len(batch_data)} lignes")
            
            # Afficher un aperçu
            st.markdown("**Aperçu des données :**")
            st.dataframe(batch_data.head(10))
            
            # Bouton de prédiction
            if st.button("🔮 Lancer les prédictions", type="primary", use_container_width=True):
                
                with st.spinner("🔄 Prédictions en cours..."):
                    # Convertir en Spark DataFrame
                    spark_df = spark.createDataFrame(batch_data)
                    
                    # Ajouter une colonne label factice
                    spark_df = spark_df.withColumn("label", col("CreditScore") * 0)
                    
                    # Faire les prédictions
                    predictions_df = model.transform(spark_df)
                    
                    # Convertir en Pandas
                    results = predictions_df.select(
                        "CreditScore", "Age", "Balance", 
                        "prediction", "probability"
                    ).toPandas()
                    
                    # Extraire les probabilités
                    results['Probabilité_Départ'] = results['probability'].apply(
                        lambda x: x[1] * 100
                    )
                    results['Prédiction'] = results['prediction'].apply(
                        lambda x: "⚠️ Risque élevé" if x == 1 else "✅ Faible risque"
                    )
                    
                    # Supprimer la colonne probability (non affichable)
                    results = results.drop(columns=['probability', 'prediction'])
                
                # Afficher les résultats
                st.markdown("---")
                st.subheader("📊 Résultats des prédictions")
                
                # Statistiques globales
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                nb_risque_eleve = (results['Prédiction'] == "⚠️ Risque élevé").sum()
                nb_risque_faible = (results['Prédiction'] == "✅ Faible risque").sum()
                taux_attrition = (nb_risque_eleve / len(results)) * 100
                
                with stats_col1:
                    st.metric("Total de clients", len(results))
                
                with stats_col2:
                    st.metric("Risque élevé", nb_risque_eleve, delta=f"{taux_attrition:.1f}%")
                
                with stats_col3:
                    st.metric("Risque faible", nb_risque_faible)
                
                # Tableau des résultats
                st.dataframe(
                    results.style.background_gradient(
                        subset=['Probabilité_Départ'], 
                        cmap='RdYlGn_r'
                    ),
                    use_container_width=True
                )
                
                # Télécharger les résultats
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les résultats (CSV)",
                    data=csv,
                    file_name="predictions_attrition.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {e}")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🏦 Application de prédiction d'attrition bancaire | 
        Modèle : Spark MLlib | 
        Interface : Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)