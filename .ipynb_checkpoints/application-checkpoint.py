import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats.mstats import winsorize

# Page Configuration
st.set_page_config(
    page_title="FEMA Disaster Analysis",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Add session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_data
def load_data():
    """Load and preprocess the FEMA disaster data."""
    try:
        file = "dataset/preprocessed/aggregate_FEMA_merged.csv"
        dates = ['declarationDate','incidentBeginDate','incidentEndDate', 'disasterCloseoutDate']
        amt = pd.read_csv(file, index_col=['disasterNumber'], parse_dates=dates, low_memory=False)
        
        amt = amt[['totalApprovedIhpAmount', 'approvedForFemaAssistance',
                  'totalInspected', 'zip_counts', 'totalMaxGrants',
                  'incidentType', 'disasterLength']]
        
        # Handle rare disaster types
        disaster_counts = amt['incidentType'].value_counts()
        rare_disasters = disaster_counts[disaster_counts < 5].index
        amt.loc[:, 'incidentType'] = amt['incidentType'].replace(rare_disasters, 'Other')
        
        # Create log-transformed DataFrame
        logamt = pd.DataFrame()
        columns_to_log = {
            'IhpAmount': 'totalApprovedIhpAmount',
            'Grants': 'totalMaxGrants',
            'FemaAssistance': 'approvedForFemaAssistance',
            'Inspected': 'totalInspected',
            'Zips': 'zip_counts',
            'Length': 'disasterLength'
        }

        for new_col, original_col in columns_to_log.items():
            logamt[new_col] = np.log1p(amt[original_col])
        logamt['Type'] = amt.incidentType

        logamt = logamt[logamt['IhpAmount'] != 0]
        logamt['IhpAmount'] = winsorize(logamt['IhpAmount'], limits=[0.02, 0.02])

        logamt = pd.get_dummies(logamt, columns=['Type'])
        for col in ['Grants', 'Inspected', 'Length']:
            logamt[col] = logamt[col].fillna(logamt[col].mean())
        return logamt, amt

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def create_summary_metrics(amt):
    """Create summary metrics for the dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Disasters", len(amt))
    with col2:
        st.metric("Average IHP Amount", f"${amt['totalApprovedIhpAmount'].mean():,.2f}")
    with col3:
        st.metric("Total ZIP Codes Affected", f"{amt['zip_counts'].sum():,}")
    with col4:
        st.metric("Total FEMA Assistance", f"{amt['approvedForFemaAssistance'].sum():,}")

def create_interactive_plots(amt):
    """Create interactive plots using Plotly"""
    # Distribution of Incident Types
    fig1 = px.pie(amt, names='incidentType', title='Distribution of Incident Types')
    st.plotly_chart(fig1)
    
    # Time series of IHP Amounts
    fig2 = px.scatter(amt, x='disasterLength', y='totalApprovedIhpAmount',
                     color='incidentType', size='zip_counts',
                     title='IHP Amount vs Disaster Length')
    st.plotly_chart(fig2)

def make_prediction(input_data, model_path):
    """Make prediction and store in history"""
    try:
        model = joblib.load(model_path)
        prediction = model.predict(input_data)
        predicted_amount = np.exp(prediction[0])
        
        # Store prediction in history
        st.session_state.prediction_history.append({
            'timestamp': pd.Timestamp.now(),
            'amount': predicted_amount,
            'model': model_path
        })
        
        return predicted_amount
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def show_prediction_history():
    """Display prediction history"""
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.subheader("Prediction History")
        st.dataframe(history_df)
        
        # Download button for history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

def test_model_tab(logamt, amt):
    """Handle test model functionality"""

    st.header("Model Testing")

    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Support Vector Machine (SVM)", "Random Forest",  "Gradient Boosting Regressor", "Linear Regression"]
    )
    tab1, tab2 = st.tabs(["Initial Assessment", "Complete Assessment"])

    with tab1:
        test_initial_assessment(logamt)

    with tab2:
        test_complete_assessment(logamt)

def prediction_tab(logamt, amt):
    """Handle prediction functionality"""
    tab1, tab2 = st.tabs(["Initial Assessment", "Complete Assessment"])

    with tab1:
        predict_initial_assessment(logamt)

    with tab2:
        predict_complete_assessment(logamt)


def test_initial_assessment(df):
    """Test the initial assessment model"""
    st.header("Initial Assessment Model")
    
    if st.button("Run Initial Assessment"):
        X = df[['Zips'] + [col for col in df.columns if 'Type_' in col]]
        model = joblib.load('SVC_model_pre-disaster.pkl')
        y = df['IhpAmount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_predict = model.predict(X_test)

        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R-squared Score", f"{r2_score(y_test, y_predict):.3f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_predict):.3f}")

        # Display results
        results = pd.DataFrame({
            "Actual": np.exp(y_test.values),
            "Predicted": np.exp(y_predict)
        })
        st.write("**Prediction Results:**")
        st.dataframe(results)

        # Create interactive plot
        fig = px.scatter(results, x="Actual", y="Predicted",
                        title="Actual vs Predicted Values")
        fig.add_trace(go.Scatter(x=[results["Actual"].min(), results["Actual"].max()],
                                y=[results["Actual"].min(), results["Actual"].max()],
                                mode='lines', name='Ideal Fit',
                                line=dict(dash='dash', color='red')))
        st.plotly_chart(fig)

def test_complete_assessment(df):
    """Test the complete assessment model"""
    st.header("Complete Assessment Model")
    
    if st.button("Run Complete Assessment"):
        X = df.drop(columns=['IhpAmount'])
        model = joblib.load('SVC_model_post-disaster.pkl')
        y = df['IhpAmount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_predict = model.predict(X_test)

        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R-squared Score", f"{r2_score(y_test, y_predict):.3f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_predict):.3f}")

        # Display results
        results = pd.DataFrame({
            "Actual": np.exp(y_test.values),
            "Predicted": np.exp(y_predict)
        })
        st.write("**Prediction Results:**")
        st.dataframe(results)

        # Create interactive plot
        fig = px.scatter(results, x="Actual", y="Predicted",
                        title="Actual vs Predicted Values")
        fig.add_trace(go.Scatter(x=[results["Actual"].min(), results["Actual"].max()],
                                y=[results["Actual"].min(), results["Actual"].max()],
                                mode='lines', name='Ideal Fit',
                                line=dict(dash='dash', color='red')))
        st.plotly_chart(fig)

def get_incident_types(logamt):
    """Extract incident types from the one-hot encoded columns"""
    type_columns = [col for col in logamt.columns if col.startswith('Type_')]
    return [col.replace('Type_', '') for col in type_columns]

def predict_initial_assessment(logamt):
    """Make predictions using the initial assessment model"""
    st.header("Initial Assessment Prediction")
    
    with st.form("initial_assessment_form"):
        st.subheader("Enter Data for Prediction")
        
        incident_types = get_incident_types(logamt)
        # Input fields
        zips = st.number_input("Number of ZIP Codes Affected", min_value=1, value=1)
        incident_type = st.selectbox("Disaster Type", incident_types)

        submitted = st.form_submit_button("Make Prediction")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'Zips': [np.log1p(zips)]
            })

            # Add dummy variables
            for type_val in incident_types:
                input_data[f'Type_{type_val}'] = 1 if type_val == incident_type else 0

            # Make prediction
            predicted_amount = make_prediction(input_data, 'SVC_model_pre-disaster.pkl')
            
            if predicted_amount is not None:
                st.success(f"Predicted IHP Amount: ${predicted_amount:,.2f}")
                
                # Add visualization
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=predicted_amount,
                    title={'text': "Predicted Amount"}
                ))
                st.plotly_chart(fig)

def predict_complete_assessment(logamt):
    """Make predictions using the complete assessment model"""
    st.header("Complete Assessment Prediction")
    
    with st.form("complete_assessment_form"):
        st.subheader("Enter Data for Prediction")

        incident_types = get_incident_types(logamt)

        col1, col2 = st.columns(2)
        
        with col1:
            fema_assistance = st.number_input("FEMA Assistance Approvals", min_value=0)
            inspected = st.number_input("Total Inspections", min_value=0)
            zips = st.number_input("Number of ZIP Codes", min_value=1, value=1)
        
        with col2:
            max_grants = st.number_input("Maximum Grants", min_value=0)
            disaster_length = st.number_input("Disaster Duration (days)", min_value=1, value=1)
            incident_type = st.selectbox("Disaster Type", incident_types)
        
        submitted = st.form_submit_button("Make Prediction")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'Grants': [np.log1p(max_grants)],
                'FemaAssistance': [np.log1p(fema_assistance)],
                'Inspected': [np.log1p(inspected)],
                'Zips': [np.log1p(zips)],
                'Length': [np.log1p(disaster_length)]
            })
            
            # Add dummy variables
            for type_val in incident_types:
                input_data[f'Type_{type_val}'] = 1 if type_val == incident_type else 0

            # Make prediction
            predicted_amount = make_prediction(input_data, 'SVC_model_post-disaster.pkl')
            
            if predicted_amount is not None:
                st.success(f"Predicted IHP Amount: ${predicted_amount:,.2f}")
                
                # Add visualization
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=predicted_amount,
                    title={'text': "Predicted Amount"}
                ))
                st.plotly_chart(fig)

def about_tab():
    st.header("About")
    st.markdown("""
    **FEMA Disaster Analysis and Prediction Tool**

    This application was developed by **Rewan Abdulkareem** in 2025 as part of the **Artificial Intelligence Olympiad**.

    ---

    ### Purpose

    The tool aims to:

    - Analyze FEMA disaster data.
    - Predict Individual and Household Program (IHP) assistance amounts using machine learning models.
    - Assist in understanding disaster impacts.
    - Aid in efficient allocation of resources during disaster response.

    ### Motivation

    Disasters have significant impacts on communities, and efficient allocation of assistance can greatly improve recovery efforts. Predicting assistance amounts can help FEMA and other agencies plan and respond more effectively.

    ### Technologies Used

    - **Python**: For data analysis and machine learning.
    - **Streamlit**: To create an interactive web application.
    - **Machine Learning Models**: Support Vector Machines (SVM) for prediction.
    - **Data Visualization**: Using Plotly and Seaborn for interactive and static plots.
    - **Data Handling**: Pandas and NumPy for data manipulation.

    """)
def main():
    st.title("🌪️ FEMA Disaster Analysis and Prediction")
    st.markdown("""
    This application analyzes FEMA disaster data and predicts Individual and Household Program (IHP) 
    assistance amounts using machine learning models.
    """)

    # Load data
    logamt, amt = load_data()
    if logamt is None or amt is None:
        st.error("Failed to load data. Please check the data source.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        mode = st.radio("Choose Mode", ["Dashboard", "Test Model", "Enter Data for Prediction", "About"])
        
    if st.checkbox("Show Data Preview"):
        st.dataframe(amt)

    if mode == "Dashboard":
        create_summary_metrics(amt)
        create_interactive_plots(amt)
        
    elif mode == "Test Model":
        test_model_tab(logamt, amt)
        
    elif mode == "Enter Data for Prediction":
        prediction_tab(logamt, amt)

    elif mode == "About":
        about_tab()
    # Show prediction history
    show_prediction_history()

    # Footer
    st.markdown("<br><hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<center>Developed by Aceel Sherif and Hana Hailaa 2025 💕</center>", unsafe_allow_html=True)
    with col2:
        st.markdown("<center>FEMA Disaster Analysis and Prediction Tool</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()