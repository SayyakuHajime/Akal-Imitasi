"""
Feature Engineering for Student Dropout Prediction
GOAL: Improve Macro-F1 Score to 72%+

Based on research insights:
- Academic performance metrics (pass rate, grade improvement, risk score)
- Economic stress indicators (financial burden, vulnerability)
- Socio-demographic factors (parent education, age risk)
- Interaction terms (academic × economic stress)

These features are CRITICAL for macro-F1 optimization because:
1. Improve Enrolled class prediction (currently bottleneck)
2. Capture non-linear relationships (failed courses + debt → high dropout)
3. Distinguish between Dropout/Enrolled/Graduate patterns

References:
- suitable_algorithm.md: Feature engineering recommendations
- prediction.md: Most important features
- other_model_insight.md: Best model achieved 90.45% macro-F1 with features
"""

import pandas as pd
import numpy as np


def create_academic_features(df):
    """
    Create academic performance metrics
    
    Key features:
    - Pass rate (most important!)
    - Grade improvement/decline
    - Academic risk score (failed courses + no-shows)
    - Average grade
    """
    df = df.copy()
    
    # Prevent division by zero
    eps = 1e-6
    
    # 1. Pass rates (CRITICAL for predicting success)
    df['pass_rate_1st'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + eps)
    df['pass_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + eps)
    df['cumulative_pass_rate'] = (
        (df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']) /
        (df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)'] + eps)
    )
    
    # 2. Grade trend (improving vs declining)
    df['grade_improvement'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    df['avg_grade'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2
    
    # 3. Failed courses
    df['failed_1st'] = df['Curricular units 1st sem (enrolled)'] - df['Curricular units 1st sem (approved)']
    df['failed_2nd'] = df['Curricular units 2nd sem (enrolled)'] - df['Curricular units 2nd sem (approved)']
    df['total_failed'] = df['failed_1st'] + df['failed_2nd']
    
    # 4. Courses without evaluation (STRONG negative indicator)
    df['total_without_eval'] = (
        df['Curricular units 1st sem (without evaluations)'] + 
        df['Curricular units 2nd sem (without evaluations)']
    )
    
    # 5. Academic risk score (combines failures + no-shows)
    # No-shows weighted 2x because they indicate disengagement
    df['academic_risk_score'] = df['total_failed'] + df['total_without_eval'] * 2
    
    # 6. Total enrolled (workload)
    df['total_enrolled'] = (
        df['Curricular units 1st sem (enrolled)'] + 
        df['Curricular units 2nd sem (enrolled)']
    )
    
    # 7. Success consistency (did well in both semesters?)
    df['consistent_performance'] = (
        (df['pass_rate_1st'] > 0.7).astype(int) * 
        (df['pass_rate_2nd'] > 0.7).astype(int)
    )
    
    return df


def create_economic_features(df):
    """
    Create economic stress indicators
    
    Key features:
    - Financial burden (debt without support)
    - Economic vulnerability (macro + personal)
    - Scholarship protection (inverse risk)
    """
    df = df.copy()
    
    # 1. Financial burden (debtor without support)
    df['financial_burden'] = (
        df['Debtor'] * 
        (1 - df['Tuition fees up to date']) * 
        (1 - df['Scholarship holder'])
    )
    
    # 2. Economic vulnerability (combines macro + personal factors)
    df['economic_vulnerability'] = (
        df['Unemployment rate'] * df['Debtor'] +
        df['Inflation rate'] * (1 - df['Scholarship holder'])
    )
    
    # 3. Scholarship protection (reduces risk)
    df['scholarship_protection'] = df['Scholarship holder'] * (1 - df['Debtor'])
    
    # 4. Payment status (up to date and no debt)
    df['financial_stability'] = df['Tuition fees up to date'] * (1 - df['Debtor'])
    
    return df


def create_sociodemographic_features(df):
    """
    Create socio-demographic risk factors
    
    Key features:
    - Parent education support
    - Support deficit (special needs without family support)
    - Age-related risk
    - Displacement burden
    """
    df = df.copy()
    
    # Prevent division by zero
    eps = 1e-6
    
    # 1. Parent education average
    df['parent_education_avg'] = (
        df["Mother's qualification"] + df["Father's qualification"]
    ) / 2
    
    # 2. Support deficit (special needs students with low parent education)
    df['support_deficit'] = (
        df['Educational special needs'] * 
        (1 / (df['parent_education_avg'] + 1 + eps))
    )
    
    # 3. Age-related risk (older students often have more responsibilities)
    df['age_risk'] = (df['Age at enrollment'] > 25).astype(int) * (df['Marital status'] + 1)
    
    # 4. Displacement burden (living away from home + financial stress)
    df['displacement_burden'] = df['Displaced'] * df['Debtor']
    
    # 5. First generation student (low parent education)
    df['first_generation'] = (
        (df["Mother's qualification"] <= 2).astype(int) * 
        (df["Father's qualification"] <= 2).astype(int)
    )
    
    return df


def create_interaction_features(df):
    """
    Create interaction terms for non-linear relationships
    
    These are CRITICAL because:
    - Dropout is often caused by COMBINATION of factors
    - E.g., failing courses is OK if you have support, but failing + debt = high risk
    """
    df = df.copy()
    
    eps = 1e-6
    
    # 1. Academic × Economic stress (MOST IMPORTANT)
    # Students failing courses AND in debt are at highest risk
    df['academic_economic_stress'] = df['academic_risk_score'] * df['financial_burden']
    
    # 2. Age × Performance (older students struggling)
    df['age_performance_risk'] = df['Age at enrollment'] * (1 - df['cumulative_pass_rate'] + eps)
    
    # 3. Grade × Financial stress
    # Assuming grades are 0-20 scale, invert to get "grade deficit"
    df['grade_financial_stress'] = (20 - df['avg_grade']) * df['financial_burden']
    
    # 4. Special needs × Performance
    df['special_needs_performance'] = df['Educational special needs'] * (1 - df['cumulative_pass_rate'] + eps)
    
    # 5. First generation × Academic risk
    df['first_gen_academic_risk'] = df['first_generation'] * df['academic_risk_score']
    
    # 6. Displacement × Academic risk
    df['displacement_academic_risk'] = df['Displaced'] * df['academic_risk_score']
    
    # 7. Age × Debt burden
    df['age_debt_burden'] = (df['Age at enrollment'] > 25).astype(int) * df['Debtor']
    
    return df


def create_enrolled_specific_features(df):
    """
    Create features specifically to distinguish ENROLLED from Dropout/Graduate
    
    Enrolled class is hardest to predict (18% minority, caught between patterns).
    These features target the "middle ground" where students are:
    - Not failing completely (like Dropout)
    - Not excelling (like Graduate)
    - Still persisting despite challenges
    
    Key patterns for Enrolled:
    - Moderate pass rate (40-80%)
    - Grade improvement trend (trying to improve)
    - Moderate academic risk (some struggles but not extreme)
    - Financial support present (helps persistence)
    - Inconsistent performance (still finding their way)
    """
    df = df.copy()
    
    eps = 1e-6
    
    # 1. Enrolled pattern: Moderate performance (not failing, not excelling)
    # Pass rate between 40-80% suggests struggling but persisting
    df['moderate_performance'] = (
        ((df['cumulative_pass_rate'] >= 0.4) & (df['cumulative_pass_rate'] <= 0.8)).astype(int)
    )
    
    # 2. Improving trajectory (key indicator of Enrolled vs Dropout)
    # Students showing improvement are likely still enrolled
    df['improving_student'] = (
        (df['grade_improvement'] >= -1).astype(int) *  # Not declining much
        (df['pass_rate_2nd'] >= df['pass_rate_1st'] - 0.1).astype(int)  # Pass rate stable or improving
    )
    
    # 3. Moderate risk (distinguishes from high-risk Dropout)
    # Academic risk score 1-4 indicates some struggles but not critical
    df['moderate_academic_risk'] = (
        ((df['academic_risk_score'] >= 1) & (df['academic_risk_score'] <= 4)).astype(int)
    )
    
    # 4. Financial support present (helps persistence despite struggles)
    # Students with scholarship OR up-to-date payments more likely to persist
    df['has_financial_support'] = (
        (df['Scholarship holder'] == 1) | (df['Tuition fees up to date'] == 1)
    ).astype(int)
    
    # 5. Inconsistent performance (still finding balance)
    # Large difference between semesters suggests adjustment period
    df['inconsistent_performance'] = (
        (abs(df['pass_rate_1st'] - df['pass_rate_2nd']) > 0.3).astype(int)
    )
    
    # 6. At-risk but persisting (key Enrolled signal)
    # Moderate risk + financial support = likely Enrolled
    df['at_risk_but_persisting'] = (
        df['moderate_academic_risk'] * df['has_financial_support']
    )
    
    # 7. Recovery pattern (failed courses but improving)
    # Had failures in 1st semester but improving in 2nd
    df['recovery_pattern'] = (
        (df['failed_1st'] >= 1).astype(int) *
        (df['grade_improvement'] > 0).astype(int) *
        (df['pass_rate_2nd'] > df['pass_rate_1st']).astype(int)
    )
    
    # 8. Middle-ground indicator (combined signal)
    # Not excelling (pass_rate < 0.9) but not failing (pass_rate > 0.3)
    # AND has some form of support
    df['enrolled_middle_ground'] = (
        df['moderate_performance'] *
        df['has_financial_support'] *
        (df['academic_risk_score'] <= 5).astype(int)
    )
    
    return df


def engineer_all_features(df):
    """
    Apply all feature engineering in correct order
    
    Returns:
        df with original + engineered features
    """
    print("Starting feature engineering...")
    original_cols = df.columns.tolist()
    
    # Step 1: Academic features (needed for interactions)
    df = create_academic_features(df)
    print(f"  Created {len(df.columns) - len(original_cols)} academic features")
    
    # Step 2: Economic features
    prev_count = len(df.columns)
    df = create_economic_features(df)
    print(f"  Created {len(df.columns) - prev_count} economic features")
    
    # Step 3: Socio-demographic features
    prev_count = len(df.columns)
    df = create_sociodemographic_features(df)
    print(f"  Created {len(df.columns) - prev_count} socio-demographic features")
    
    # Step 4: Interaction features (requires features from steps 1-3)
    prev_count = len(df.columns)
    df = create_interaction_features(df)
    print(f"  Created {len(df.columns) - prev_count} interaction features")
    
    # Step 5: Enrolled-specific features (NEW - requires features from steps 1-4)
    prev_count = len(df.columns)
    df = create_enrolled_specific_features(df)
    print(f"  Created {len(df.columns) - prev_count} enrolled-specific features")
    
    # Handle any NaN or inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with 0 (reasonable for engineered features)
    engineered_cols = [col for col in df.columns if col not in original_cols]
    df[engineered_cols] = df[engineered_cols].fillna(0)
    
    print(f"\nFeature engineering complete!")
    print(f"Total features: {len(df.columns)} (original: {len(original_cols)}, engineered: {len(engineered_cols)})")
    
    return df


def get_feature_importance_groups():
    """
    Return feature groups for analysis
    """
    return {
        'academic': [
            'pass_rate_1st', 'pass_rate_2nd', 'cumulative_pass_rate',
            'grade_improvement', 'avg_grade',
            'failed_1st', 'failed_2nd', 'total_failed',
            'total_without_eval', 'academic_risk_score',
            'total_enrolled', 'consistent_performance'
        ],
        'economic': [
            'financial_burden', 'economic_vulnerability',
            'scholarship_protection', 'financial_stability'
        ],
        'sociodemographic': [
            'parent_education_avg', 'support_deficit',
            'age_risk', 'displacement_burden', 'first_generation'
        ],
        'interactions': [
            'academic_economic_stress', 'age_performance_risk',
            'grade_financial_stress', 'special_needs_performance',
            'first_gen_academic_risk', 'displacement_academic_risk',
            'age_debt_burden'
        ],
        'enrolled_specific': [
            'moderate_performance', 'improving_student',
            'moderate_academic_risk', 'has_financial_support',
            'inconsistent_performance', 'at_risk_but_persisting',
            'recovery_pattern', 'enrolled_middle_ground'
        ]
    }


if __name__ == '__main__':
    # Test feature engineering
    print("="*80)
    print("TESTING FEATURE ENGINEERING")
    print("="*80)
    
    # Load training data
    train_df = pd.read_csv('data/train.csv')
    print(f"\nOriginal training data shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()[:5]}... (showing first 5)")
    
    # Engineer features
    train_engineered = engineer_all_features(train_df)
    
    # Show new features
    original_cols = train_df.columns.tolist()
    new_cols = [col for col in train_engineered.columns if col not in original_cols]
    
    print(f"\nNew engineered features ({len(new_cols)}):")
    for group_name, features in get_feature_importance_groups().items():
        print(f"\n{group_name.upper()}:")
        for feat in features:
            if feat in new_cols:
                print(f"  - {feat}")
    
    # Show statistics
    print("\n" + "="*80)
    print("ENGINEERED FEATURE STATISTICS")
    print("="*80)
    print(train_engineered[new_cols].describe())
    
    # Save for inspection
    output_file = 'data/train_engineered_sample.csv'
    train_engineered.head(100).to_csv(output_file, index=False)
    print(f"\nSaved sample (first 100 rows) to: {output_file}")
    
    print("\nFeature engineering ready for model training!")
