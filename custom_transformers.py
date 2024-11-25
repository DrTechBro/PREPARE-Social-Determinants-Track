from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Mappings for ordinal variables
age_mapping = {
    '0. 49 or younger': 0,
    '1. 50–59': 1,
    '2. 60–69': 2,
    '3. 70–79': 3,
    '4. 80+': 4,
}

education_mapping = {
    '0. No education': 0,
    '1. 1–5 years': 1,
    '2. 6 years': 2,
    '3. 7–9 years': 3,
    '4. 10+ years': 4,
}

n_living_child_mapping = {
    '0. No children': 0,
    '1. 1 or 2': 1,
    '2. 3 or 4': 2,
    '3. 5 or 6': 3,
    '4. 7+': 4,
}

glob_health_mapping = {
    '1. Excellent': 5,
    '2. Very good': 4,
    '3. Good': 3,
    '4. Fair': 2,
    '5. Poor': 1,
}

bmi_mapping = {
    '1. Underweight': 1,
    '2. Normal weight': 2,
    '3. Overweight': 3,
    '4. Obese': 4,
    '5. Morbidly obese': 5,
}

decis_famil_mapping = {
    '1. Respondent': 1,
    '2. Approximately equal weight': 2,
    '3. Spouse': 3,
}

decis_personal_mapping = {
    '1. A lot': 3,
    '2. A little': 2,
    '3. None': 1
}

agreement_mapping = {
    '1. Agrees': 3,
    '2. Neither agrees nor disagrees': 2,
    '3. Disagrees': 1,
}

memory_mapping = {
    '1. Excellent': 5,
    '2. Very good': 4,
    '3. Good': 3,
    '4. Fair': 2,
    '5. Poor': 1,
}

parent_education_mapping = {
    '1.None': 1,
    '2.Some primary': 2,
    '3.Primary': 3,
    '4.More than primary': 4,
}

religion_importance_mapping = {
    '1.very important': 3,
    '2.somewhat important': 2,
    '3.not important': 1,
}

frequency_mapping = {
    '1.Almost every day': 9,
    '2.4 or more times a week': 8,
    '3.2 or 3 times a week': 7,
    '4.Once a week': 6,
    '5.4 or more times a month': 5,
    '6.2 or 3 times a month': 4,
    '7.Once a month': 3,
    '8.Almost Never, sporadic': 2,
    '9.Never': 1,
}

religious_services_mapping = {
    '1.Yes': 1,
    '0.No': 0,
}

english_proficiency_mapping = {
    'Yes 1': 1,
    'No 2': 0,
}

# Compile all mappings into a dictionary for easy access
ordinal_mappings = {
    'age_03': age_mapping,
    'age_12': age_mapping,
    'edu_gru_03': education_mapping,
    'edu_gru_12': education_mapping,
    'n_living_child_03': n_living_child_mapping,
    'n_living_child_12': n_living_child_mapping,
    'glob_hlth_03': glob_health_mapping,
    'glob_hlth_12': glob_health_mapping,
    'bmi_12': bmi_mapping,
    'decis_famil_12': decis_famil_mapping,
    'decis_personal_12': decis_personal_mapping,
    'satis_ideal_12': agreement_mapping,
    'satis_excel_12': agreement_mapping,
    'satis_fine_12': agreement_mapping,
    'cosas_imp_12': agreement_mapping,
    'wouldnt_change_12': agreement_mapping,
    'memory_12': memory_mapping,
    'rameduc_m': parent_education_mapping,
    'rafeduc_m': parent_education_mapping,
    'rrelgimp_03': religion_importance_mapping,
    'rrelgimp_12': religion_importance_mapping,
    'rrfcntx_m_12': frequency_mapping,
    'rsocact_m_12': frequency_mapping,
    'rrelgwk_12': religious_services_mapping,
    'a34_12': english_proficiency_mapping,
}

# Function to map ordinal variables
def map_ordinal_variables(X, ordinal_cols, ordinal_mappings):
    X = X.copy()
    for col in ordinal_cols:
        if col in X.columns:
            mapping = ordinal_mappings.get(col, {})
            X[col] = X[col].map(mapping)
    return X

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates temporal features such as change over time
    and time gap since last measurement.
    """
    def __init__(self, numerical_common_features, ordinal_mappings):
        self.numerical_common_features = numerical_common_features
        self.ordinal_mappings = ordinal_mappings
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names for get_feature_names_out()
        self.feature_names_in_ = X.columns
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        
        # Determine last feature year using vectorized operations
        X['last_feature_year'] = self.get_last_feature_year(X)

        # Calculate time gap between last measurement and target year
        if 'year' in X.columns and 'last_feature_year' in X.columns:
            X['time_gap'] = X['year'] - X['last_feature_year']
        else:
            X['time_gap'] = np.nan  # Handle missing 'year' or 'last_feature_year'
        
        # Handle ordinal variables and map them
        for feature in self.numerical_common_features:
            base_feature = feature  # e.g., 'age', 'edu_gru'
            col_03 = feature + '_03'
            col_12 = feature + '_12'
            change_col = feature + '_change'

            # Map ordinal variables if necessary
            if base_feature in self.ordinal_mappings:
                mapping = self.ordinal_mappings[base_feature]
                if col_03 in X.columns:
                    X[col_03] = X[col_03].map(mapping)
                if col_12 in X.columns:
                    X[col_12] = X[col_12].map(mapping)

            # Convert columns to numeric (if not already)
            if col_03 in X.columns:
                X[col_03] = pd.to_numeric(X[col_03], errors='coerce')
            if col_12 in X.columns:
                X[col_12] = pd.to_numeric(X[col_12], errors='coerce')

            # Calculate change: 2012 value - 2003 value
            if col_03 in X.columns and col_12 in X.columns:
                # Both columns exist
                valid_mask = X[col_03].notnull() & X[col_12].notnull()
                X.loc[valid_mask, change_col] = X.loc[valid_mask, col_12] - X.loc[valid_mask, col_03]
                X.loc[~valid_mask, change_col] = np.nan
            else:
                X[change_col] = np.nan  # Set change to NaN if both columns do not exist

        return X

    def get_feature_names_out(self, input_features=None):
        # Return the feature names after transformation
        # Include new features added during transform
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)

        # Add new features
        output_features.append('last_feature_year')
        output_features.append('time_gap')

        # Add change columns
        for feature in self.numerical_common_features:
            change_col = feature + '_change'
            output_features.append(change_col)

        return np.array(output_features)

    @staticmethod
    def get_last_feature_year(X):
        """
        Determine the last year when data is available for each individual using vectorized operations.
        """
        data_2012_cols = [col for col in X.columns if col.endswith('_12')]
        data_2003_cols = [col for col in X.columns if col.endswith('_03')]

        has_2012_data = X[data_2012_cols].notnull().any(axis=1)
        has_2003_data = X[data_2003_cols].notnull().any(axis=1)

        last_year = pd.Series(np.nan, index=X.index)
        last_year[has_2012_data] = 2012
        last_year[~has_2012_data & has_2003_data] = 2003

        return last_year


class EducationProgressionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the change in education level between 2003 and 2012.
    """
    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        if 'edu_gru_03' in X.columns and 'edu_gru_12' in X.columns:
            # Ensure ordinal mappings have been applied
            X['edu_gru_03'] = pd.to_numeric(X['edu_gru_03'], errors='coerce')
            X['edu_gru_12'] = pd.to_numeric(X['edu_gru_12'], errors='coerce')
            
            valid_mask = X['edu_gru_03'].notnull() & X['edu_gru_12'].notnull()
            X.loc[valid_mask, 'education_transition'] = X.loc[valid_mask, 'edu_gru_12'] - X.loc[valid_mask, 'edu_gru_03']
            X.loc[~valid_mask, 'education_transition'] = np.nan
        else:
            X['education_transition'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including the new feature
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.append('education_transition')
        return np.array(output_features)

class MaritalTransitionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the marital transition between two time points.
    """
    def __init__(self, married_cols_03, married_cols_12):
        self.married_cols_03 = married_cols_03
        self.married_cols_12 = married_cols_12
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure columns exist
        if self.married_cols_03[0] in X.columns and self.married_cols_12[0] in X.columns:
            # Convert to numeric if necessary
            X[self.married_cols_03[0]] = pd.to_numeric(X[self.married_cols_03[0]], errors='coerce')
            X[self.married_cols_12[0]] = pd.to_numeric(X[self.married_cols_12[0]], errors='coerce')
            # Calculate marital transition
            X['marital_transition'] = X[self.married_cols_12[0]] - X[self.married_cols_03[0]]
        else:
            X['marital_transition'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including the new feature
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.append('marital_transition')
        return np.array(output_features)

# Count of chronic illnesses, with change over time
class ChronicIllnessTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the total number of chronic illnesses and change over time.
    """
    def __init__(self, chronic_illness_cols_03, chronic_illness_cols_12):
        self.chronic_illness_cols_03 = chronic_illness_cols_03
        self.chronic_illness_cols_12 = chronic_illness_cols_12
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        illness_cols_03 = [col for col in self.chronic_illness_cols_03 if col in X.columns]
        illness_cols_12 = [col for col in self.chronic_illness_cols_12 if col in X.columns]
        
        # Convert to numeric and fill missing values
        if illness_cols_03:
            X[illness_cols_03] = X[illness_cols_03].apply(pd.to_numeric, errors='coerce').fillna(0)
            X['chronic_illness_count_03'] = X[illness_cols_03].sum(axis=1)
        else:
            X['chronic_illness_count_03'] = np.nan
            
        if illness_cols_12:
            X[illness_cols_12] = X[illness_cols_12].apply(pd.to_numeric, errors='coerce').fillna(0)
            X['chronic_illness_count_12'] = X[illness_cols_12].sum(axis=1)
        else:
            X['chronic_illness_count_12'] = np.nan

        # Calculate change over time if both counts are available
        if 'chronic_illness_count_03' in X.columns and 'chronic_illness_count_12' in X.columns:
            valid_mask = X['chronic_illness_count_03'].notnull() & X['chronic_illness_count_12'].notnull()
            X.loc[valid_mask, 'chronic_illness_count_change'] = X.loc[valid_mask, 'chronic_illness_count_12'] - X.loc[valid_mask, 'chronic_illness_count_03']
            X.loc[~valid_mask, 'chronic_illness_count_change'] = np.nan
        else:
            X['chronic_illness_count_change'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including new features
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.extend(['chronic_illness_count_03', 'chronic_illness_count_12', 'chronic_illness_count_change'])
        return np.array(output_features)

# Limitations of Activities of daily living count and progression
class ADLIADLTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the total number of ADL and IADL limitations and progression.
    """
    def __init__(self, adl_cols_03, adl_cols_12, iadl_cols_03, iadl_cols_12):
        self.adl_cols_03 = adl_cols_03
        self.adl_cols_12 = adl_cols_12
        self.iadl_cols_03 = iadl_cols_03
        self.iadl_cols_12 = iadl_cols_12
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        # Check if required columns are present
        adl_cols_present = all(col in X.columns for col in self.adl_cols_03 + self.adl_cols_12)
        iadl_cols_present = all(col in X.columns for col in self.iadl_cols_03 + self.iadl_cols_12)
        if adl_cols_present and iadl_cols_present:
            # Calculate total ADL limitations
            X['total_adl_limitations_03'] = X[self.adl_cols_03].sum(axis=1)
            X['total_adl_limitations_12'] = X[self.adl_cols_12].sum(axis=1)
            # Calculate total IADL limitations
            X['total_iadl_limitations_03'] = X[self.iadl_cols_03].sum(axis=1)
            X['total_iadl_limitations_12'] = X[self.iadl_cols_12].sum(axis=1)
            # Calculate progression
            X['adl_iadl_progression'] = (
                (X['total_adl_limitations_12'] + X['total_iadl_limitations_12']) -
                (X['total_adl_limitations_03'] + X['total_iadl_limitations_03'])
            )
        else:
            X['adl_iadl_progression'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including new features
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.extend([
            'total_adl_limitations_03', 'total_adl_limitations_12',
            'total_iadl_limitations_03', 'total_iadl_limitations_12',
            'adl_iadl_progression'
        ])
        return np.array(output_features)

# Self Reported Health Change
class HealthAssessmentChangeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the change in self-reported global health status.
    """
    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        if 'glob_hlth_03' in X.columns and 'glob_hlth_12' in X.columns:
            # Ensure that both columns are numeric (assuming mapping is done)
            X['glob_hlth_03'] = pd.to_numeric(X['glob_hlth_03'], errors='coerce')
            X['glob_hlth_12'] = pd.to_numeric(X['glob_hlth_12'], errors='coerce')
            
            # Identify valid entries where both health assessments are not null
            valid_mask = X['glob_hlth_03'].notnull() & X['glob_hlth_12'].notnull()
            X.loc[valid_mask, 'health_self_assessment_change'] = X.loc[valid_mask, 'glob_hlth_12'] - X.loc[valid_mask, 'glob_hlth_03']
            X.loc[~valid_mask, 'health_self_assessment_change'] = np.nan
        else:
            X['health_self_assessment_change'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including the new feature
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.append('health_self_assessment_change')
        return np.array(output_features)

# Custom transformer to engineer positive and negative mood scores
class MoodScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes aggregate positive and negative mood scores and changes.
    """
    def __init__(self, positive_mood_cols_03, positive_mood_cols_12, negative_mood_cols_03, negative_mood_cols_12):
        self.positive_mood_cols_03 = positive_mood_cols_03
        self.positive_mood_cols_12 = positive_mood_cols_12
        self.negative_mood_cols_03 = negative_mood_cols_03
        self.negative_mood_cols_12 = negative_mood_cols_12
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Adjust columns based on those present in X
        pos_mood_cols_03 = [col for col in self.positive_mood_cols_03 if col in X.columns]
        neg_mood_cols_03 = [col for col in self.negative_mood_cols_03 if col in X.columns]
        pos_mood_cols_12 = [col for col in self.positive_mood_cols_12 if col in X.columns]
        neg_mood_cols_12 = [col for col in self.negative_mood_cols_12 if col in X.columns]
        
        # Ensure columns are numeric
        for cols in [pos_mood_cols_03, neg_mood_cols_03, pos_mood_cols_12, neg_mood_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
        
        # Fill missing values with 0
        for cols in [pos_mood_cols_03, neg_mood_cols_03, pos_mood_cols_12, neg_mood_cols_12]:
            if cols:
                X[cols] = X[cols].fillna(0)
        
        # Create aggregate scores for positive and negative moods
        if pos_mood_cols_03:
            X['positive_mood_score_03'] = X[pos_mood_cols_03].sum(axis=1)
        else:
            X['positive_mood_score_03'] = np.nan
        
        if neg_mood_cols_03:
            X['negative_mood_score_03'] = X[neg_mood_cols_03].sum(axis=1)
        else:
            X['negative_mood_score_03'] = np.nan
        
        if pos_mood_cols_12:
            X['positive_mood_score_12'] = X[pos_mood_cols_12].sum(axis=1)
        else:
            X['positive_mood_score_12'] = np.nan
        
        if neg_mood_cols_12:
            X['negative_mood_score_12'] = X[neg_mood_cols_12].sum(axis=1)
        else:
            X['negative_mood_score_12'] = np.nan
        
        # Calculate mood changes over time
        valid_pos = X['positive_mood_score_03'].notnull() & X['positive_mood_score_12'].notnull()
        X['positive_mood_change'] = np.nan
        X.loc[valid_pos, 'positive_mood_change'] = X.loc[valid_pos, 'positive_mood_score_12'] - X.loc[valid_pos, 'positive_mood_score_03']
        
        valid_neg = X['negative_mood_score_03'].notnull() & X['negative_mood_score_12'].notnull()
        X['negative_mood_change'] = np.nan
        X.loc[valid_neg, 'negative_mood_change'] = X.loc[valid_neg, 'negative_mood_score_12'] - X.loc[valid_neg, 'negative_mood_score_03']
        
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including new features
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        new_features = [
            'positive_mood_score_03', 'negative_mood_score_03',
            'positive_mood_score_12', 'negative_mood_score_12',
            'positive_mood_change', 'negative_mood_change'
        ]
        output_features.extend(new_features)
        return np.array(output_features)

# Consistency of exercise tracking
class ConsistentExerciseTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates a feature indicating consistent exercise.
    """
    def __init__(self):
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self  # Nothing to fit
    
    def transform(self, X):
        X = X.copy()
        if 'exer_3xwk_03' in X.columns and 'exer_3xwk_12' in X.columns:
            # Convert to numeric
            X['exer_3xwk_03'] = pd.to_numeric(X['exer_3xwk_03'], errors='coerce')
            X['exer_3xwk_12'] = pd.to_numeric(X['exer_3xwk_12'], errors='coerce')
            
            # Identify valid entries
            valid_mask = X['exer_3xwk_03'].notnull() & X['exer_3xwk_12'].notnull()
            
            # Initialize the feature with NaN
            X['consistent_exercise'] = np.nan
            
            # Compute consistent_exercise
            X.loc[valid_mask, 'consistent_exercise'] = (
                (X.loc[valid_mask, 'exer_3xwk_03'] == 1) & (X.loc[valid_mask, 'exer_3xwk_12'] == 1)
            ).astype(int)
        else:
            X['consistent_exercise'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including 'consistent_exercise'
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.append('consistent_exercise')
        return np.array(output_features)

# Alcohol and smoking tracking
class LifestyleHealthIndexTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes a lifestyle health index and changes over time.
    """
    def __init__(self, lifestyle_cols_03, lifestyle_cols_12):
        self.lifestyle_cols_03 = lifestyle_cols_03
        self.lifestyle_cols_12 = lifestyle_cols_12
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        lifestyle_cols_03 = [col for col in self.lifestyle_cols_03 if col in X.columns]
        lifestyle_cols_12 = [col for col in self.lifestyle_cols_12 if col in X.columns]

        # Ensure columns are numeric and fill missing values
        for cols in [lifestyle_cols_03, lifestyle_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Sum lifestyle health indices
        if lifestyle_cols_03:
            X['lifestyle_health_index_03'] = X[lifestyle_cols_03].sum(axis=1)
        else:
            X['lifestyle_health_index_03'] = np.nan

        if lifestyle_cols_12:
            X['lifestyle_health_index_12'] = X[lifestyle_cols_12].sum(axis=1)
        else:
            X['lifestyle_health_index_12'] = np.nan

        # Calculate change over time
        valid_mask = X['lifestyle_health_index_03'].notnull() & X['lifestyle_health_index_12'].notnull()
        X['lifestyle_health_index_change'] = np.nan
        X.loc[valid_mask, 'lifestyle_health_index_change'] = X.loc[valid_mask, 'lifestyle_health_index_12'] - X.loc[valid_mask, 'lifestyle_health_index_03']

        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including new features
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.extend([
            'lifestyle_health_index_03', 'lifestyle_health_index_12', 'lifestyle_health_index_change'
        ])
        return np.array(output_features)

# Income and insurance coverage
class SocioeconomicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes aggregate income, insurance coverage depth, and insurance continuity.
    """
    def __init__(self, income_cols_03, income_cols_12, insurance_cols_03, insurance_cols_12):
        self.income_cols_03 = income_cols_03
        self.income_cols_12 = income_cols_12
        self.insurance_cols_03 = insurance_cols_03
        self.insurance_cols_12 = insurance_cols_12
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        income_cols_03 = [col for col in self.income_cols_03 if col in X.columns]
        income_cols_12 = [col for col in self.income_cols_12 if col in X.columns]
        insurance_cols_03 = [col for col in self.insurance_cols_03 if col in X.columns]
        insurance_cols_12 = [col for col in self.insurance_cols_12 if col in X.columns]
        
        # Convert to numeric and fill missing values
        for cols in [income_cols_03, income_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        for cols in [insurance_cols_03, insurance_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Aggregate income
        if income_cols_03:
            X['aggregate_income_03'] = X[income_cols_03].sum(axis=1)
        else:
            X['aggregate_income_03'] = np.nan
        if income_cols_12:
            X['aggregate_income_12'] = X[income_cols_12].sum(axis=1)
        else:
            X['aggregate_income_12'] = np.nan
        
        # Insurance coverage depth
        if insurance_cols_03:
            X['insurance_coverage_depth_03'] = X[insurance_cols_03].sum(axis=1)
        else:
            X['insurance_coverage_depth_03'] = np.nan
        if insurance_cols_12:
            X['insurance_coverage_depth_12'] = X[insurance_cols_12].sum(axis=1)
        else:
            X['insurance_coverage_depth_12'] = np.nan
        
        # Insurance continuity
        valid_insurance = X['insurance_coverage_depth_03'].notnull() & X['insurance_coverage_depth_12'].notnull()
        X['insurance_continuity'] = np.nan
        X.loc[valid_insurance, 'insurance_continuity'] = ((X.loc[valid_insurance, 'insurance_coverage_depth_03'] > 0) & (X.loc[valid_insurance, 'insurance_coverage_depth_12'] > 0)).astype(int)
        
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including new features
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        new_features = [
            'aggregate_income_03', 'aggregate_income_12',
            'insurance_coverage_depth_03', 'insurance_coverage_depth_12',
            'insurance_continuity'
        ]
        output_features.extend(new_features)
        return np.array(output_features)


# Social Engagement score
class SocialEngagementTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes a social engagement score for each individual in 2012.
    """
    def __init__(self, social_engagement_cols):
        self.social_engagement_cols = social_engagement_cols
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        social_engagement_cols = [col for col in self.social_engagement_cols if col in X.columns]
        
        # Ensure columns are numeric and fill missing values
        if social_engagement_cols:
            X[social_engagement_cols] = X[social_engagement_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Sum social engagement score
            X['social_engagement_12'] = X[social_engagement_cols].sum(axis=1)
        else:
            X['social_engagement_12'] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including 'social_engagement_12'
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        output_features.append('social_engagement_12')
        return np.array(output_features)

# Custom transformer to create preventive care index and health services usage
class HealthServicesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates indices for preventive care and health service usage.
    """
    def __init__(self, preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12):
        self.preventive_care_cols_03 = preventive_care_cols_03
        self.preventive_care_cols_12 = preventive_care_cols_12
        self.health_service_usage_cols_03 = health_service_usage_cols_03
        self.health_service_usage_cols_12 = health_service_usage_cols_12
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        preventive_care_cols_03 = [col for col in self.preventive_care_cols_03 if col in X.columns]
        preventive_care_cols_12 = [col for col in self.preventive_care_cols_12 if col in X.columns]
        health_service_usage_cols_03 = [col for col in self.health_service_usage_cols_03 if col in X.columns]
        health_service_usage_cols_12 = [col for col in self.health_service_usage_cols_12 if col in X.columns]

        # Convert to numeric and fill missing values
        for cols in [preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Create preventive care index for 2003 and 2012
        if preventive_care_cols_03:
            X['preventive_care_index_03'] = X[preventive_care_cols_03].sum(axis=1)
        else:
            X['preventive_care_index_03'] = np.nan

        if preventive_care_cols_12:
            X['preventive_care_index_12'] = X[preventive_care_cols_12].sum(axis=1)
        else:
            X['preventive_care_index_12'] = np.nan

        # Create health service usage score for 2003 and 2012
        if health_service_usage_cols_03:
            X['health_service_usage_03'] = X[health_service_usage_cols_03].sum(axis=1)
        else:
            X['health_service_usage_03'] = np.nan

        if health_service_usage_cols_12:
            X['health_service_usage_12'] = X[health_service_usage_cols_12].sum(axis=1)
        else:
            X['health_service_usage_12'] = np.nan

        # Calculate changes between 2003 and 2012
        valid_preventive = X['preventive_care_index_03'].notnull() & X['preventive_care_index_12'].notnull()
        X['preventive_care_change'] = np.nan
        X.loc[valid_preventive, 'preventive_care_change'] = X.loc[valid_preventive, 'preventive_care_index_12'] - X.loc[valid_preventive, 'preventive_care_index_03']

        valid_usage = X['health_service_usage_03'].notnull() & X['health_service_usage_12'].notnull()
        X['health_service_usage_change'] = np.nan
        X.loc[valid_usage, 'health_service_usage_change'] = X.loc[valid_usage, 'health_service_usage_12'] - X.loc[valid_usage, 'health_service_usage_03']

        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including new features
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        new_features = [
            'preventive_care_index_03', 'preventive_care_index_12', 'preventive_care_change',
            'health_service_usage_03', 'health_service_usage_12', 'health_service_usage_change'
        ]
        output_features.extend(new_features)
        return np.array(output_features)

# Create a custom transformer that applies all custom transformations
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        numerical_common_features,
        ordinal_mappings,
        married_cols_03,
        married_cols_12,
        chronic_illness_cols_03,
        chronic_illness_cols_12,
        adl_cols_03,
        adl_cols_12,
        iadl_cols_03,
        iadl_cols_12,
        positive_mood_cols_03,
        positive_mood_cols_12,
        negative_mood_cols_03,
        negative_mood_cols_12,
        lifestyle_cols_03,
        lifestyle_cols_12,
        income_cols_03,
        income_cols_12,
        insurance_cols_03,
        insurance_cols_12,
        social_engagement_cols,
        preventive_care_cols_03,
        preventive_care_cols_12,
        health_service_usage_cols_03,
        health_service_usage_cols_12
    ):
        # Assign parameters to instance variables
        self.numerical_common_features = numerical_common_features
        self.ordinal_mappings = ordinal_mappings
        self.married_cols_03 = married_cols_03
        self.married_cols_12 = married_cols_12
        self.chronic_illness_cols_03 = chronic_illness_cols_03
        self.chronic_illness_cols_12 = chronic_illness_cols_12
        self.adl_cols_03 = adl_cols_03
        self.adl_cols_12 = adl_cols_12
        self.iadl_cols_03 = iadl_cols_03
        self.iadl_cols_12 = iadl_cols_12
        self.positive_mood_cols_03 = positive_mood_cols_03
        self.positive_mood_cols_12 = positive_mood_cols_12
        self.negative_mood_cols_03 = negative_mood_cols_03
        self.negative_mood_cols_12 = negative_mood_cols_12
        self.lifestyle_cols_03 = lifestyle_cols_03
        self.lifestyle_cols_12 = lifestyle_cols_12
        self.income_cols_03 = income_cols_03
        self.income_cols_12 = income_cols_12
        self.insurance_cols_03 = insurance_cols_03
        self.insurance_cols_12 = insurance_cols_12
        self.social_engagement_cols = social_engagement_cols
        self.preventive_care_cols_03 = preventive_care_cols_03
        self.preventive_care_cols_12 = preventive_care_cols_12
        self.health_service_usage_cols_03 = health_service_usage_cols_03
        self.health_service_usage_cols_12 = health_service_usage_cols_12
        self.feature_names_in_ = None

        # Initialize all custom transformers using the instance attributes
        self.temporal_features = TemporalFeatureEngineer(self.numerical_common_features, self.ordinal_mappings)
        self.education_progression = EducationProgressionTransformer()
        self.marital_transition = MaritalTransitionTransformer(self.married_cols_03, self.married_cols_12)
        self.chronic_illness = ChronicIllnessTransformer(self.chronic_illness_cols_03, self.chronic_illness_cols_12)
        self.adl_iadl = ADLIADLTransformer(self.adl_cols_03, self.adl_cols_12, self.iadl_cols_03, self.iadl_cols_12)
        self.health_assessment_change = HealthAssessmentChangeTransformer()
        self.mood_score = MoodScoreTransformer(self.positive_mood_cols_03, self.positive_mood_cols_12, self.negative_mood_cols_03, self.negative_mood_cols_12)
        self.consistent_exercise = ConsistentExerciseTransformer()
        self.lifestyle_health_index = LifestyleHealthIndexTransformer(self.lifestyle_cols_03, self.lifestyle_cols_12)
        self.socioeconomic_features = SocioeconomicFeaturesTransformer(self.income_cols_03, self.income_cols_12, self.insurance_cols_03, self.insurance_cols_12)
        self.social_engagement = SocialEngagementTransformer(self.social_engagement_cols)
        self.health_services = HealthServicesTransformer(self.preventive_care_cols_03, self.preventive_care_cols_12, self.health_service_usage_cols_03, self.health_service_usage_cols_12)
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        # Fit all transformers (if necessary)
        self.temporal_features.fit(X, y)
        self.education_progression.fit(X, y)
        self.marital_transition.fit(X, y)
        self.chronic_illness.fit(X, y)
        self.adl_iadl.fit(X, y)
        self.health_assessment_change.fit(X, y)
        self.mood_score.fit(X, y)
        self.consistent_exercise.fit(X, y)
        self.lifestyle_health_index.fit(X, y)
        self.socioeconomic_features.fit(X, y)
        self.social_engagement.fit(X, y)
        self.health_services.fit(X, y)
        return self
    
    def transform(self, X):
        X = X.copy()
        X = self.temporal_features.transform(X)
        X = self.education_progression.transform(X)
        X = self.marital_transition.transform(X)
        X = self.chronic_illness.transform(X)
        X = self.adl_iadl.transform(X)
        X = self.health_assessment_change.transform(X)
        X = self.mood_score.transform(X)
        X = self.consistent_exercise.transform(X)
        X = self.lifestyle_health_index.transform(X)
        X = self.socioeconomic_features.transform(X)
        X = self.social_engagement.transform(X)
        X = self.health_services.transform(X)
        return X

    def get_feature_names_out(self, input_features=None):
        # Aggregate feature names from all transformers
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = input_features

        # Collect feature names from each transformer
        transformers = [
            self.temporal_features, self.education_progression, self.marital_transition,
            self.chronic_illness, self.adl_iadl, self.health_assessment_change,
            self.mood_score, self.consistent_exercise, self.lifestyle_health_index,
            self.socioeconomic_features, self.social_engagement, self.health_services
        ]

        for transformer in transformers:
            output_features = transformer.get_feature_names_out(output_features)

        return output_features

    

class InteractionTermsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates interaction terms.
    """
    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        
        # Interaction: Health service usage change * Lifestyle health index (2012)
        if 'health_service_usage_change' in X.columns and 'lifestyle_health_index_12' in X.columns:
            X['health_lifestyle_interaction'] = (
                X['health_service_usage_change'] * X['lifestyle_health_index_12']
            )
        
        # Interaction: Education transition * Aggregate income (2012)
        if 'education_transition' in X.columns and 'aggregate_income_12' in X.columns:
            X['education_income_interaction'] = (
                X['education_transition'] * X['aggregate_income_12']
            )
        
        # Interaction: Social engagement (2012) * Positive mood change
        if 'social_engagement_12' in X.columns and 'positive_mood_change' in X.columns:
            X['social_mood_interaction'] = (
                X['social_engagement_12'] * X['positive_mood_change']
            )
        
        # Interaction: Preventive care change * Chronic illness count (2012)
        if 'preventive_care_change' in X.columns and 'chronic_illness_count_12' in X.columns:
            X['preventive_chronic_interaction'] = (
                X['preventive_care_change'] * X['chronic_illness_count_12']
            )
        
        return X

    def get_feature_names_out(self, input_features=None):
        # Return feature names including interaction terms
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        interaction_features = [
            'health_lifestyle_interaction', 'education_income_interaction',
            'social_mood_interaction', 'preventive_chronic_interaction'
        ]
        output_features.extend(interaction_features)
        return np.array(output_features)

# SHAP-based feature selection as a custom transformer
class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects top features based on SHAP values.
    """
    def __init__(self, base_model, num_features):
        self.base_model = base_model
        self.num_features = num_features
        self.selected_features = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        self.feature_names_in_ = X.columns
        # Train the base model for SHAP analysis
        self.base_model.fit(X, y)
        explainer = shap.TreeExplainer(self.base_model)
        shap_values = explainer.shap_values(X)
        # Calculate mean absolute SHAP values for feature importance
        feature_importances = np.abs(shap_values).mean(axis=0)
        # Select top features
        top_indices = np.argsort(feature_importances)[-self.num_features:]
        self.selected_features = X.columns[top_indices]
        return self

    def transform(self, X):
        # Return only the selected features
        return X[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        # Return the names of the selected features
        return self.selected_features


# Ordinal Mapper Transformer
class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_cols, ordinal_mappings):
        self.ordinal_cols = ordinal_cols
        self.ordinal_mappings = ordinal_mappings
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store the feature names for later use
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.ordinal_cols:
            if col in X.columns:
                mapping = self.ordinal_mappings.get(col, {})
                X[col] = X[col].map(mapping)
        return X

    def get_feature_names_out(self, input_features=None):
        # Return the feature names after transformation
        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

class CustomPipeline(Pipeline):
    def get_feature_names_out(self, input_features=None):
        features = input_features
        for name, transform in self.steps:
            if hasattr(transform, 'get_feature_names_out'):
                if features is None:
                    features = transform.get_feature_names_out()
                else:
                    features = transform.get_feature_names_out(features)
            else:
                pass  # Transformer does not support get_feature_names_out
        return features
