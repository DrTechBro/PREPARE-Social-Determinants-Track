import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from custom_transformers import TemporalFeatureEngineer, EducationProgressionTransformer, MaritalTransitionTransformer, ChronicIllnessTransformer, ADLIADLTransformer, HealthAssessmentChangeTransformer, MoodScoreTransformer, ConsistentExerciseTransformer, LifestyleHealthIndexTransformer, SocioeconomicFeaturesTransformer, SocialEngagementTransformer, HealthServicesTransformer, CustomFeatureEngineer, InteractionTermsTransformer, SHAPFeatureSelector, OrdinalMapper
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Set matplotlib to display plots inline
# st.set_option('deprecation.showPyplotGlobalUse', False)

def map_ordinal_variables(X, ordinal_cols, ordinal_mappings):
    """Function to map ordinal variables to numerical values."""
    X = X.copy()
    for col in ordinal_cols:
        if col in X.columns:
            mapping = ordinal_mappings.get(col, {})
            X[col] = X[col].map(mapping)
    return X

# Load the pre-trained model
try:
    model = joblib.load("stacked_model.pkl")
except FileNotFoundError:
    st.error("The model file 'stacked_model.pkl' was not found. Ensure the file is in the same directory as this script.")

# Function to preprocess data
def preprocess_data(test_data, expected_features):
    """Preprocess the test data for prediction."""
    # Drop unnecessary columns
    features_to_drop = ["uid"]
    X_test = test_data.drop(columns=features_to_drop, errors="ignore")

    # Reindex to match training features
    X_test = X_test.reindex(columns=expected_features, fill_value=0)

    return X_test

# Streamlit app configuration
st.set_page_config(
    page_title="Cognitive Insights Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# **Header Section**
st.title("🧠 Cognitive Insights Hub")
st.markdown(
    """
    #### Harnessing the Power of Data Science to Advance Alzheimer's Research  
    This app predicts **composite cognitive scores** to aid early detection of Alzheimer's Disease, 
    part of the **PREPARE Project** leveraging social determinants of health (SDOH).  
    """
)
st.markdown("---")

# **Sidebar Section**
st.sidebar.header("Upload Your Files")
st.sidebar.markdown(
    """
    **Instructions**:  
    1. Upload the **Submission Format** file (CSV).  
    2. Upload the **Test Features** file (CSV).  
    3. View detailed predictions and insights.
    """
)

# File uploaders
submission_file = st.file_uploader("Upload Submission Format File", type="csv")
features_file = st.file_uploader("Upload Test Features File", type="csv")

if submission_file is not None and features_file is not None:
    try:
        # Load the submission format and test features files
        submission_format = pd.read_csv(submission_file)
        test_features = pd.read_csv(features_file)

        st.write("Submission Format Preview:")
        st.write(submission_format.head())

        st.write("Test Features Preview:")
        st.write(test_features.head())

        # Merge the submission format with test features
        test_data = submission_format[["uid", "year"]].merge(test_features, on=["uid"], how="left")

        # Prepare features
        X_test = test_data.drop(columns=["uid"], errors="ignore")

        # Defining the expected features
        expected_features = ['year', 'age_03', 'urban_03', 'married_03', 'n_mar_03', 'edu_gru_03', 'n_living_child_03', 'migration_03', 'glob_hlth_03', 'adl_dress_03', 'adl_walk_03', 'adl_bath_03', 'adl_eat_03', 'adl_bed_03', 'adl_toilet_03', 'n_adl_03', 'iadl_money_03', 'iadl_meds_03', 'iadl_shop_03', 'iadl_meals_03', 'n_iadl_03', 'depressed_03', 'hard_03', 'restless_03', 'happy_03', 'lonely_03', 'enjoy_03', 'sad_03', 'tired_03', 'energetic_03', 'n_depr_03', 'cesd_depressed_03', 'hypertension_03', 'diabetes_03', 'resp_ill_03', 'arthritis_03', 'hrt_attack_03', 'stroke_03', 'cancer_03', 'n_illnesses_03', 'exer_3xwk_03', 'alcohol_03', 'tobacco_03', 'test_chol_03', 'test_tuber_03', 'test_diab_03', 'test_pres_03', 'hosp_03', 'visit_med_03', 'out_proc_03', 'visit_dental_03', 'imss_03', 'issste_03', 'pem_def_mar_03', 'insur_private_03', 'insur_other_03', 'insured_03', 'decis_personal_03', 'employment_03', 'age_12', 'urban_12', 'married_12', 'n_mar_12', 'edu_gru_12', 'n_living_child_12', 'migration_12', 'glob_hlth_12', 'adl_dress_12', 'adl_walk_12', 'adl_bath_12', 'adl_eat_12', 'adl_bed_12', 'adl_toilet_12', 'n_adl_12', 'iadl_money_12', 'iadl_meds_12', 'iadl_shop_12', 'iadl_meals_12', 'n_iadl_12', 'depressed_12', 'hard_12', 'restless_12', 'happy_12', 'lonely_12', 'enjoy_12', 'sad_12', 'tired_12', 'energetic_12', 'n_depr_12', 'cesd_depressed_12', 'hypertension_12', 'diabetes_12', 'resp_ill_12', 'arthritis_12', 'hrt_attack_12', 'stroke_12', 'cancer_12', 'n_illnesses_12', 'bmi_12', 'exer_3xwk_12', 'alcohol_12', 'tobacco_12', 'test_chol_12', 'test_tuber_12', 'test_diab_12', 'test_pres_12', 'hosp_12', 'visit_med_12', 'out_proc_12', 'visit_dental_12', 'imss_12', 'issste_12', 'pem_def_mar_12', 'insur_private_12', 'insur_other_12', 'insured_12', 'decis_famil_12', 'decis_personal_12', 'employment_12', 'vax_flu_12', 'vax_pneu_12', 'seg_pop_12', 'care_adult_12', 'care_child_12', 'volunteer_12', 'attends_class_12', 'attends_club_12', 'reads_12', 'games_12', 'table_games_12', 'comms_tel_comp_12', 'act_mant_12', 'tv_12', 'sewing_12', 'satis_ideal_12', 'satis_excel_12', 'satis_fine_12', 'cosas_imp_12', 'wouldnt_change_12', 'memory_12', 'ragender', 'rameduc_m', 'rafeduc_m', 'sgender_03', 'rearnings_03', 'searnings_03', 'hincome_03', 'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03', 'hinc_cap_03', 'rinc_pension_03', 'sinc_pension_03', 'rrelgimp_03', 'sgender_12', 'rjlocc_m_12', 'rearnings_12', 'searnings_12', 'hincome_12', 'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12', 'hinc_cap_12', 'rinc_pension_12', 'sinc_pension_12', 'rrelgimp_12', 'rrfcntx_m_12', 'rsocact_m_12', 'rrelgwk_12', 'a34_12', 'j11_12']
    
        # Preprocess the data
        preprocessed_data = preprocess_data(test_data, expected_features)

        # Generate predictions
        with st.spinner("Predicting cognitive scores..."):
            y_pred = model.predict(preprocessed_data)

        # Round and convert predictions to integer
        y_pred_int = np.round(y_pred).astype(int)

        # Add predictions to the original test data
        test_data["composite_score"] = y_pred_int

        # Reorder columns to place uid, year, and composite_score first
        reordered_columns = ["uid", "year", "composite_score"] + [col for col in test_data.columns if col not in ["uid", "year", "composite_score"]]
        test_data = test_data[reordered_columns]

        # Display predictions and features
        st.markdown("### Prediction Results")
        st.write(test_data)

        # Download button
        st.download_button(
            label="📥 Download Detailed Predictions",
            data=test_data.to_csv(index=False),
            file_name="detailed_predictions.csv",
            mime="text/csv",
        )

        # **Summary Section**
        st.markdown("### Summary Insights")
        avg_score = test_data["composite_score"].mean()
        st.metric(label="Average Composite Score", value=f"{avg_score:.2f}")

        # Display histogram
        st.markdown("#### Distribution of Predicted Scores")
        st.bar_chart(test_data["composite_score"].value_counts().sort_index())

        

    except Exception as e:
        st.error(f"🚨 An error occurred: {e}")
        st.exception(e)
else:
    st.info("Please upload both files to proceed.")

# Footer
st.markdown("---")
st.markdown(
    """
    *Developed with ❤️ for advancing Alzheimer's research.*  
    [Learn More About the PREPARE Project](https://www.drivendata.org/competitions/300/competition-nih-alzheimers-sdoh-2/page/928/)
    """
)