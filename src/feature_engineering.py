# Feature Engineering utilities for Student Dropout Prediction
# Based on analysis in .private/suitable_algorithm.md

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create academic performance metrics.
    
    Features created:
    - cumulative_success_rate: (approved courses) / (enrolled courses)
    - grade_trend: 2nd semester grade - 1st semester grade
    - academic_risk: combines failures, missing evaluations, and low grades
    """
    
    def __init__(self, sem1_approved_col='Curricular units 1st sem (approved)',
                 sem2_approved_col='Curricular units 2nd sem (approved)',
                 sem1_enrolled_col='Curricular units 1st sem (enrolled)',
                 sem2_enrolled_col='Curricular units 2nd sem (enrolled)',
                 sem1_grade_col='Curricular units 1st sem (grade)',
                 sem2_grade_col='Curricular units 2nd sem (grade)',
                 sem1_without_eval_col='Curricular units 1st sem (without evaluations)',
                 sem2_without_eval_col='Curricular units 2nd sem (without evaluations)'):
        
        self.sem1_approved_col = sem1_approved_col
        self.sem2_approved_col = sem2_approved_col
        self.sem1_enrolled_col = sem1_enrolled_col
        self.sem2_enrolled_col = sem2_enrolled_col
        self.sem1_grade_col = sem1_grade_col
        self.sem2_grade_col = sem2_grade_col
        self.sem1_without_eval_col = sem1_without_eval_col
        self.sem2_without_eval_col = sem2_without_eval_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Cumulative success rate
        total_approved = X_new[self.sem1_approved_col] + X_new[self.sem2_approved_col]
        total_enrolled = X_new[self.sem1_enrolled_col] + X_new[self.sem2_enrolled_col]
        X_new['cumulative_success_rate'] = np.where(
            total_enrolled > 0,
            total_approved / total_enrolled,
            0
        )
        
        # Grade trend (improvement or decline)
        X_new['grade_trend'] = X_new[self.sem2_grade_col] - X_new[self.sem1_grade_col]
        
        # Academic risk score (higher = more risk)
        # Combines: missing evaluations + (20 - grade) to normalize grade scale
        X_new['academic_risk'] = (
            X_new[self.sem1_without_eval_col] + X_new[self.sem2_without_eval_col] +
            (20 - X_new[self.sem1_grade_col]) + (20 - X_new[self.sem2_grade_col])
        )
        
        # Failure rate per semester
        X_new['sem1_failure_rate'] = np.where(
            X_new[self.sem1_enrolled_col] > 0,
            (X_new[self.sem1_enrolled_col] - X_new[self.sem1_approved_col]) / X_new[self.sem1_enrolled_col],
            0
        )
        X_new['sem2_failure_rate'] = np.where(
            X_new[self.sem2_enrolled_col] > 0,
            (X_new[self.sem2_enrolled_col] - X_new[self.sem2_approved_col]) / X_new[self.sem2_enrolled_col],
            0
        )
        
        return X_new


class EconomicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create economic stress indicators.
    
    Features created:
    - financial_burden: combination of debtor status, tuition fees, and scholarship
    - economic_vulnerability: macro economic factors × personal financial status
    """
    
    def __init__(self, debtor_col='Debtor',
                 tuition_col='Tuition fees up to date',
                 scholarship_col='Scholarship holder',
                 unemployment_col='Unemployment rate',
                 inflation_col='Inflation rate'):
        
        self.debtor_col = debtor_col
        self.tuition_col = tuition_col
        self.scholarship_col = scholarship_col
        self.unemployment_col = unemployment_col
        self.inflation_col = inflation_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Financial burden (higher = worse financial situation)
        # debtor=1 AND tuition not up to date=0 AND no scholarship=0
        X_new['financial_burden'] = (
            X_new[self.debtor_col] * 
            (1 - X_new[self.tuition_col]) * 
            (1 - X_new[self.scholarship_col])
        )
        
        # Economic vulnerability (macro + personal)
        # High unemployment + debtor OR high inflation + no scholarship
        if self.unemployment_col in X_new.columns and self.inflation_col in X_new.columns:
            X_new['economic_vulnerability'] = (
                X_new[self.unemployment_col] * X_new[self.debtor_col] +
                X_new[self.inflation_col] * (1 - X_new[self.scholarship_col])
            )
        
        return X_new


class SocioDemographicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create socio-demographic risk indicators.
    
    Features created:
    - parent_education_avg: average education level of parents
    - support_deficit: special needs without adequate family support
    - maturity_risk: older students with marital responsibilities
    """
    
    def __init__(self, mother_qual_col="Mother's qualification",
                 father_qual_col="Father's qualification",
                 special_needs_col='Educational special needs',
                 age_col='Age at enrollment',
                 marital_col='Marital status',
                 displaced_col='Displaced'):
        
        self.mother_qual_col = mother_qual_col
        self.father_qual_col = father_qual_col
        self.special_needs_col = special_needs_col
        self.age_col = age_col
        self.marital_col = marital_col
        self.displaced_col = displaced_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Parent education average (proxy for family support)
        if self.mother_qual_col in X_new.columns and self.father_qual_col in X_new.columns:
            X_new['parent_education_avg'] = (
                X_new[self.mother_qual_col] + X_new[self.father_qual_col]
            ) / 2
            
            # Support deficit: special needs with low parental education
            if self.special_needs_col in X_new.columns:
                X_new['support_deficit'] = (
                    X_new[self.special_needs_col] / (X_new['parent_education_avg'] + 1)
                )
        
        # Maturity risk: older students (>25) with marital responsibilities
        if self.age_col in X_new.columns and self.marital_col in X_new.columns:
            X_new['maturity_risk'] = (X_new[self.age_col] > 25).astype(int) * X_new[self.marital_col]
        
        return X_new


class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create interaction features (critical for non-linear relationships).
    
    Features created:
    - academic_economic_stress: academic risk × financial burden
    - age_performance: age × (1 - success rate)
    - displacement_burden: displaced × debtor
    """
    
    def __init__(self, required_features=None):
        """
        Parameters:
        -----------
        required_features : list of str or None
            List of features that must exist before creating interactions.
            If None, will create interactions from available features.
        """
        self.required_features = required_features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Academic × Economic interaction
        if 'academic_risk' in X_new.columns and 'financial_burden' in X_new.columns:
            X_new['academic_economic_stress'] = (
                X_new['academic_risk'] * X_new['financial_burden']
            )
        
        # Age × Performance interaction
        if 'Age at enrollment' in X_new.columns and 'cumulative_success_rate' in X_new.columns:
            X_new['age_performance'] = (
                X_new['Age at enrollment'] * (1 - X_new['cumulative_success_rate'])
            )
        
        # Displaced × Financial interaction
        if 'Displaced' in X_new.columns and 'Debtor' in X_new.columns:
            X_new['displacement_burden'] = X_new['Displaced'] * X_new['Debtor']
        
        # Grade trend × Financial burden
        if 'grade_trend' in X_new.columns and 'financial_burden' in X_new.columns:
            X_new['declining_performance_stress'] = np.where(
                X_new['grade_trend'] < 0,  # Grade declining
                abs(X_new['grade_trend']) * X_new['financial_burden'],
                0
            )
        
        return X_new


class CompleteFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Apply all feature engineering steps in sequence.
    This is a convenience class that combines all feature engineering transformers.
    """
    
    def __init__(self):
        self.academic_engineer = AcademicFeatureEngineer()
        self.economic_engineer = EconomicFeatureEngineer()
        self.socio_engineer = SocioDemographicFeatureEngineer()
        self.interaction_engineer = InteractionFeatureEngineer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Apply transformations in sequence
        X_new = self.academic_engineer.transform(X_new)
        X_new = self.economic_engineer.transform(X_new)
        X_new = self.socio_engineer.transform(X_new)
        X_new = self.interaction_engineer.transform(X_new)
        
        return X_new
    
    def get_feature_names(self):
        """Return list of all engineered feature names."""
        return [
            # Academic features
            'cumulative_success_rate',
            'grade_trend',
            'academic_risk',
            'sem1_failure_rate',
            'sem2_failure_rate',
            # Economic features
            'financial_burden',
            'economic_vulnerability',
            # Socio-demographic features
            'parent_education_avg',
            'support_deficit',
            'maturity_risk',
            # Interaction features
            'academic_economic_stress',
            'age_performance',
            'displacement_burden',
            'declining_performance_stress'
        ]


# Example usage:
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'Curricular units 1st sem (approved)': [5, 3, 6],
        'Curricular units 2nd sem (approved)': [4, 2, 6],
        'Curricular units 1st sem (enrolled)': [6, 6, 6],
        'Curricular units 2nd sem (enrolled)': [6, 6, 6],
        'Curricular units 1st sem (grade)': [12, 10, 15],
        'Curricular units 2nd sem (grade)': [13, 9, 16],
        'Curricular units 1st sem (without evaluations)': [0, 2, 0],
        'Curricular units 2nd sem (without evaluations)': [1, 3, 0],
        'Debtor': [0, 1, 0],
        'Scholarship holder': [1, 0, 1],
        'Tuition fees up to date': [1, 0, 1],
        'Age at enrollment': [18, 35, 20],
        'Marital status': [0, 1, 0],
        'Displaced': [0, 1, 0],
        "Mother's qualification": [5, 2, 6],
        "Father's qualification": [4, 3, 5],
        'Educational special needs': [0, 1, 0],
        'Unemployment rate': [10.5, 10.5, 10.5],
        'Inflation rate': [2.3, 2.3, 2.3]
    })
    
    # Apply feature engineering
    engineer = CompleteFeatureEngineer()
    transformed = engineer.fit_transform(sample_data)
    
    print("Original shape:", sample_data.shape)
    print("Transformed shape:", transformed.shape)
    print("\nNew features created:")
    for feat in engineer.get_feature_names():
        if feat in transformed.columns:
            print(f"  - {feat}")
