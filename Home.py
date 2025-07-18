import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
st.set_page_config(layout="wide", page_title="Health Insurance Cost Predictor")

sex_mapping = {'male': 0, 'female': 1}
smoker_mapping = {'no': 0, 'yes': 1}
region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

bmi_bins = [0, 18.5, 25, 30, 55]
bmi_labels = ['underweight', 'normal', 'overweight', 'obese']

age_bins = [0, 30, 50, 100]
age_labels = ['young', 'middle-aged', 'senior']

reverse_sex_mapping = {v: k for k, v in sex_mapping.items()}
reverse_smoker_mapping = {v: k for k, v in smoker_mapping.items()}
reverse_region_mapping = {v: k for k, v in region_mapping.items()}


@st.cache_data
def load_data():
    try:
        health_insurance = pd.read_csv('insurance.csv')
    except FileNotFoundError:
        st.error("`insurance.csv` not found. Please make sure the file is in the same directory.")
        st.stop()

    health_insurance['sex'] = health_insurance['sex'].map(sex_mapping)
    health_insurance['smoker'] = health_insurance['smoker'].map(smoker_mapping)
    health_insurance['region'] = health_insurance['region'].map(region_mapping)

    health_insurance['bmi_category'] = pd.cut(health_insurance['bmi'], bins=bmi_bins, labels=bmi_labels)
    health_insurance['age_group'] = pd.cut(health_insurance['age'], bins=age_bins, labels=age_labels)

    return health_insurance


@st.cache_resource
def load_model_and_test_data():
    try:
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        with open('y_test.pkl', 'rb') as file:
            y_test = pickle.load(file)
    except FileNotFoundError:
        st.error("Model files (`random_forest_model.pkl`, `X_test.pkl`, `y_test.pkl`) not found.")
        st.info("Please ensure these pickle files are in the same directory as this Streamlit app.")
        st.stop()

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, X_test, y_test, r2, mse, mae


df = load_data()
model, X_test, y_test, r2, mse, mae = load_model_and_test_data()

display_df = df.copy()
display_df['sex_label'] = display_df['sex'].map(reverse_sex_mapping)
display_df['smoker_label'] = display_df['smoker'].map(reverse_smoker_mapping)
display_df['region_label'] = display_df['region'].map(reverse_region_mapping)

st.sidebar.title("Explore the App")
page_selection = st.sidebar.radio("Go to", ["Home", "About", "Data Insights", "Model Performance"])
st.sidebar.markdown("---")

# Initialize session state for metrics if not already present
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'history': []
    }

# Home page (Prediction)
if page_selection == "Home":
    st.title("Health Insurance Cost Predictor")
    st.markdown("""
        Welcome to the **Health Insurance Cost Predictor**
        This application allows you to estimate health insurance charges based on personal attributes.
        Enter your details to get an instant prediction.
    """)

    st.header("User Input Features")


    def get_user_input():
        age = st.number_input("Age", min_value=18, max_value=64, value=30, step=1)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1, format="%.1f")
        children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, step=1)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

        sex_encoded = sex_mapping[sex]
        smoker_encoded = smoker_mapping[smoker]
        region_encoded = region_mapping[region]

        user_bmi_category = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels)[0]
        user_age_group = pd.cut([age], bins=age_bins, labels=age_labels)[0]

        user_data = {
            'age': age,
            'sex': sex_encoded,
            'bmi': bmi,
            'children': children,
            'smoker': smoker_encoded,
            'region': region_encoded
        }
        features = pd.DataFrame(user_data, index=[0])

        expected_columns = list(X_test.columns)
        for col in expected_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[expected_columns]

        return features, user_bmi_category, user_age_group


    user_input_df, user_bmi_category, user_age_group = get_user_input()

    st.subheader("Your Predicted Insurance Charges")
    predicted_charges = model.predict(user_input_df)
    st.metric(label="Estimated Charges", value=f"${predicted_charges[0]:,.2f}")

    st.markdown(f"**Based on your input:**")
    st.markdown(f"- **BMI Category:** `{user_bmi_category}`")
    st.markdown(f"- **Age Group:** `{user_age_group}`")


    simulated_actual = predicted_charges[0] * np.random.uniform(0.9, 1.1)

    # Update the test data and metrics
    new_X_test = pd.concat([X_test, user_input_df], ignore_index=True)
    new_y_test = pd.concat([y_test, pd.Series([simulated_actual])], ignore_index=True)

    new_pred = model.predict(new_X_test)
    new_r2 = r2_score(new_y_test, new_pred)
    new_mse = mean_squared_error(new_y_test, new_pred)
    new_mae = mean_absolute_error(new_y_test, new_pred)

    # Store in session state
    st.session_state.metrics = {
        'r2': new_r2,
        'mse': new_mse,
        'mae': new_mae,
        'history': st.session_state.metrics['history'] + [{
            'input': user_input_df.iloc[0].to_dict(),
            'predicted': predicted_charges[0],
            'simulated_actual': simulated_actual,
            'metrics': {
                'r2': new_r2,
                'mse': new_mse,
                'mae': new_mae
            }
        }]
    }

# About page
elif page_selection == "About":
    st.title("About the Health Insurance Cost Predictor")
    st.markdown("""
        ### Welcome!
        This application is designed to provide an **estimated prediction** of individual health insurance charges based on several key personal attributes. It leverages a machine learning model trained on a real-world dataset to offer insights into potential insurance costs.

        ### How It Works
        1.  **Input Your Details:** On the **Home** page, use the interactive sidebar to input your age, sex, BMI, number of children, smoking status, and region.
        2.  **Instant Prediction:** As you adjust your details, the app will instantly display an estimated insurance charge based on the trained model.
        3.  **Explore Insights:**
            * Visit the **Data Insights** page to understand the underlying dataset, view distributions of features, and discover trends.
            * Check the **Model Performance** page to see how well our machine learning model performs and its key evaluation metrics.

        ### Why This App?
        Understanding factors that influence health insurance costs can be complex. This tool aims to provide a transparent and interactive way to explore these relationships and get a quick estimate. """)
    st.markdown("---")
    st.subheader("Meet the Developer")
    st.write(
        "This application was developed as a demonstration of machine learning for predictive analytics. You can find more projects and connect with me on [LinkedIn](https://www.linkedin.com/in/aseel-ai-ml/) or [GitHub](https://github.com/Aseel92).")

# Data Insights page
elif page_selection == "Data Insights":
    st.title("Data Insights and Exploration")
    st.markdown("""This page provides a deeper look into the dataset used to train the insurance cost prediction model.
        Explore distributions, relationships between features, and the categorical mappings that was used in the Feature Engineering Phase.""")
    st.markdown("---")

    st.subheader("Dataset Overview")
    st.dataframe(display_df.head())

    st.write("#### Basic Descriptive Statistics")
    st.dataframe(df.describe())

    st.markdown("---")

    st.subheader("Mapping Used for Categorical Features")
    st.write("For numerical processing, categorical features are converted to numbers as follows:")

    st.write("#### Sex Mapping")
    st.json(sex_mapping)

    st.write("#### Smoker Mapping")
    st.json(smoker_mapping)

    st.write("#### Region Mapping")
    st.json(region_mapping)

    st.markdown("---")

    st.subheader("Key Visualizations from the Analysis")

    # 1. Histograms for Numerical Features
    st.write("#### Distribution of Key Numerical Features")
    fig_hist, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df['age'], kde=True, bins=30, color='skyblue', ax=axes[0])
    axes[0].set_title('Age Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')

    sns.histplot(df['bmi'], kde=True, bins=30, color='lightcoral', ax=axes[1])
    axes[1].set_title('BMI Distribution')
    axes[1].set_xlabel('BMI')
    axes[1].set_ylabel('Frequency')

    sns.histplot(df['charges'], kde=True, bins=30, color='lightgreen', ax=axes[2])
    axes[2].set_title('Charges Distribution')
    axes[2].set_xlabel('Charges')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig_hist)
    st.markdown("---")

    # 2. BMI Boxplot - Modified to show colors by bmi_category
    st.write("#### BMI Distribution by Category")
    fig_bmi_cat, ax_bmi_cat = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='bmi_category', y='bmi', data=df, ax=ax_bmi_cat, palette="viridis", order=bmi_labels)
    ax_bmi_cat.set_title('BMI Distribution by Category', fontsize=16)
    ax_bmi_cat.set_xlabel('BMI Category', fontsize=14)
    ax_bmi_cat.set_ylabel('BMI', fontsize=14)
    st.pyplot(fig_bmi_cat)
    st.markdown("---")

    # 3. Charges Boxplot
    st.write("#### Charges Distribution")
    fig_charges, ax_charges = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df['charges'], ax=ax_charges, palette="plasma")
    ax_charges.set_title('Charges Distribution', fontsize=16)
    ax_charges.set_ylabel('Charges', fontsize=14)
    st.pyplot(fig_charges)
    st.markdown("---")

    # 4. Smoker vs. Charges Boxplot
    st.write("#### Charges by Smoker Status")
    fig_smoker_charges, ax_smoker_charges = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='smoker_label', y='charges', data=display_df, ax=ax_smoker_charges, palette="coolwarm")
    ax_smoker_charges.set_title('Charges by Smoker Status', fontsize=16)
    ax_smoker_charges.set_xlabel('Smoker', fontsize=14)
    ax_smoker_charges.set_ylabel('Charges', fontsize=14)
    st.pyplot(fig_smoker_charges)
    st.markdown("---")

    # 5. Age vs. Charges by Smoker Status
    st.write("#### Age vs. Charges by Smoker Status")
    fig_age_charges_smoker, ax_age_charges_smoker = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='age', y='charges', hue='smoker_label', data=display_df, palette='coolwarm', alpha=0.7,
                    ax=ax_age_charges_smoker)
    ax_age_charges_smoker.set_title('Age vs. Charges by Smoker Status', fontsize=16)
    ax_age_charges_smoker.set_xlabel('Age', fontsize=14)
    ax_age_charges_smoker.set_ylabel('Charges', fontsize=14)
    ax_age_charges_smoker.legend(title='Smoker')
    st.pyplot(fig_age_charges_smoker)
    st.markdown("---")

    # 6. BMI vs. Charges by Smoker Status
    st.write("#### BMI vs. Charges by Smoker Status")
    fig_bmi_charges_smoker, ax_bmi_charges_smoker = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='bmi', y='charges', hue='smoker_label', data=display_df, palette='viridis', alpha=0.7,
                    ax=ax_bmi_charges_smoker)
    ax_bmi_charges_smoker.set_title('BMI vs. Charges by Smoker Status', fontsize=16)
    ax_bmi_charges_smoker.set_xlabel('BMI', fontsize=14)
    ax_bmi_charges_smoker.set_ylabel('Charges', fontsize=14)
    ax_bmi_charges_smoker.legend(title='Smoker')
    st.pyplot(fig_bmi_charges_smoker)
    st.markdown("---")

    # 7. Countplot for Gender Distribution
    st.write("#### Count of Individuals by Sex")
    fig_sex_count, ax_sex_count = plt.subplots(figsize=(10, 7))
    sns.countplot(x='sex_label', data=display_df, palette='coolwarm', ax=ax_sex_count)
    ax_sex_count.set_title('Count of Individuals by Sex', fontsize=16)
    ax_sex_count.set_xlabel('Sex', fontsize=14)
    ax_sex_count.set_ylabel('Count', fontsize=14)
    st.pyplot(fig_sex_count)
    st.markdown("---")

    # 8. Countplot for Children Distribution
    st.write("#### Count of Individuals by Number of Children")
    fig_children_count, ax_children_count = plt.subplots(figsize=(8, 6))
    sns.countplot(x='children', data=display_df, palette='magma', ax=ax_children_count)
    ax_children_count.set_title('Count of Individuals by Number of Children', fontsize=16)
    ax_children_count.set_xlabel('Number of Children', fontsize=14)
    ax_children_count.set_ylabel('Count', fontsize=14)
    st.pyplot(fig_children_count)
    st.markdown("---")

    # 9. Countplot for Region Distribution
    st.write("#### Count of Individuals by Region")
    fig_region_count, ax_region_count = plt.subplots(figsize=(9, 6))
    sns.countplot(x='region_label', data=display_df, palette='viridis', order=list(region_mapping.keys()),
                  ax=ax_region_count)
    ax_region_count.set_title('Count of Individuals by Region', fontsize=16)
    ax_region_count.set_xlabel('Region', fontsize=14)
    ax_region_count.set_ylabel('Count', fontsize=14)
    st.pyplot(fig_region_count)
    st.markdown("---")



# Model Performance Page
elif page_selection == "Model Performance":
    st.title("Model Performance Evaluation")
    st.markdown("""
        This page presents the key performance metrics of the machine learning model used for prediction.
        Understanding these metrics helps in evaluating the model's accuracy and reliability.
    """)
    st.markdown("---")

    # Use session state metrics
    current_metrics = st.session_state.metrics

    st.subheader("Current Performance Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="R-squared (R2) Score", value=f"{current_metrics['r2']:.4f}")
        st.markdown(
            "Higher is better. Explains the proportion of variance in the dependent variable that can be predicted from the independent variables.")
    with col2:
        st.metric(label="Mean Squared Error (MSE)", value=f"{current_metrics['mse']:,.2f}")
        st.markdown(
            "Lower is better. Represents the average of the squared differences between predicted and actual values.")
    with col3:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{current_metrics['mae']:,.2f}")
        st.markdown(
            "Lower is better. Represents the average of the absolute differences between predicted and actual values, in the same units as the target variable ($).")

    st.markdown("---")



    st.subheader("Actual vs. Predicted Charges")
    st.write(
        "This scatter plot visualizes how well the model's predictions align with the actual insurance charges. A perfect model would have all points lying on the red dashed line.")

    # Get predictions for the test set
    y_pred = model.predict(X_test)

    fig_pred, ax_pred = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue', ax=ax_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax_pred.set_title('Actual vs. Predicted Charges', fontsize=16)
    ax_pred.set_xlabel('Actual Values', fontsize=14)
    ax_pred.set_ylabel('Predicted Values', fontsize=14)
    st.pyplot(fig_pred)
    st.markdown("---")

