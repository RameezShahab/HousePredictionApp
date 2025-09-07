import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import joblib
import sys
import os

# Add the current directory to path to import project_script if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Ames Housing Explorer", layout="wide")

st.title("🏠 Ames Housing — Price Prediction App")
st.write("Explore the dataset and predict house prices using machine learning models.")

# Load data - use relative path for Streamlit Cloud
DATA_PATH = "AmesHousing.csv"

# File uploader
uploaded = st.file_uploader("Upload a CSV (or leave blank to use the included AmesHousing.csv)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded CSV loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded CSV: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv(DATA_PATH)
        st.success("✅ Default dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read {DATA_PATH}: {e}. Please upload a CSV file.")
        st.stop()

# Display dataset info
st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
st.sidebar.write(f"**Columns:** {len(df.columns)}")
st.sidebar.write(f"**Numeric columns:** {len(df.select_dtypes(include=[np.number]).columns)}")
st.sidebar.write(f"**Categorical columns:** {len(df.select_dtypes(include=['object']).columns)}")

# Sidebar controls
st.sidebar.header("🔍 Data Explorer")

# Show raw data checkbox
if st.sidebar.checkbox("📋 Show raw data", value=False):
    st.header("Raw Data Preview")
    st.dataframe(df.head(200))

st.sidebar.subheader("📈 Column Explorer")
columns = df.columns.tolist()
selected_col = st.sidebar.selectbox("Select column to inspect", columns, index=0)
st.header(f"Column Analysis: {selected_col}")

# Show column statistics
col_data = df[selected_col]
if pd.api.types.is_numeric_dtype(col_data):
    st.subheader("Numeric Statistics")
    col_stats = col_data.describe()
    st.write(col_stats)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution")
        st.line_chart(col_data.dropna().reset_index(drop=True))
    
    with col2:
        st.subheader("Histogram")
        hist_chart = alt.Chart(df).mark_bar().encode(
            alt.X(selected_col, bin=True),
            y='count()'
        )
        st.altair_chart(hist_chart, use_container_width=True)
else:
    st.subheader("Categorical Distribution")
    value_counts = col_data.value_counts().head(20)
    st.write(value_counts)
    
    # Bar chart for categorical data
    bar_chart = alt.Chart(df).mark_bar().encode(
        x=selected_col,
        y='count()'
    ).properties(height=300)
    st.altair_chart(bar_chart, use_container_width=True)

# Scatter Plot Section
st.sidebar.subheader("📊 Scatter Plot")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        pair_x = st.selectbox("X axis", numeric_cols, index=0)
    with col2:
        # Try to find a different column for Y axis
        y_index = 1 if len(numeric_cols) > 1 else 0
        pair_y = st.selectbox("Y axis", numeric_cols, index=y_index)
    
    if pair_x and pair_y and pair_x != pair_y:
        st.header(f"Scatter Plot: {pair_x} vs {pair_y}")
        
        # Filter out infinite values
        plot_df = df[[pair_x, pair_y]].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(plot_df) > 0:
            chart = alt.Chart(plot_df).mark_circle(size=60).encode(
                x=alt.X(pair_x, title=pair_x),
                y=alt.Y(pair_y, title=pair_y),
                tooltip=[pair_x, pair_y]
            ).interactive().properties(
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No valid data points to plot for the selected columns.")
    else:
        st.info("Please select two different numeric columns for the scatter plot.")
else:
    st.sidebar.warning("Need at least 2 numeric columns for scatter plot")

# Model Prediction Section
st.sidebar.header("🤖 Model Prediction")
st.sidebar.write("Train machine learning models on the dataset")

if st.sidebar.button("🚀 Train Models", type="primary"):
    st.header("Model Training Results")
    
    # Import and run your training script
    try:
        # Import your training script components
        from project_script import models, train_test_split
        import numpy as np
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare the data (assuming SalePrice is the target)
        if 'SalePrice' in df.columns:
            X = df.drop('SalePrice', axis=1)
            y = df['SalePrice']
            
            # Split the data
            train_df, test_df, train_target, test_target = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results = []
            
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    # Fit the model
                    model.fit(train_df, train_target)
                    
                    # Predict on test set
                    preds = model.predict(test_df)
                    
                    # Evaluate the model
                    rmse = np.sqrt(mean_squared_error(test_target, preds))
                    r2 = r2_score(test_target, preds)
                    
                    results.append((name, rmse, r2))
            
            # Display results in a nice table
            st.subheader("📊 Model Performance Comparison")
            
            # Create a results dataframe
            results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'R² Score'])
            results_df = results_df.sort_values('RMSE')
            
            # Display the table
            st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='lightgreen')
                                 .highlight_max(subset=['R² Score'], color='lightblue'))
            
            # Show best model
            best_model = results_df.iloc[0]
            st.success(f"🎯 Best Model: {best_model['Model']} (RMSE: {best_model['RMSE']:.2f}, R²: {best_model['R² Score']:.3f})")
            
        else:
            st.error("❌ 'SalePrice' column not found in the dataset. Please make sure your dataset contains the target variable.")
            
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        st.info("💡 Make sure your project_script.py is properly configured and all required columns are present.")

# Footer
st.markdown("---")
st.markdown("""
**🔧 Next steps to enhance this app**:
1. Add interactive prediction interface
2. Save trained models for faster loading
3. Add feature importance visualizations
4. Include data preprocessing options

**📁 Files in this project**: `app.py`, `project_script.py`, `AmesHousing.csv`, `requirements.txt`
""")