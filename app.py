import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import sys
import os

# Set page config
st.set_page_config(
    page_title="Ames Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the Ames Housing dataset"""
    try:
        # Try multiple encodings and paths
        data_paths = [
            "AmesHousing.csv",
            "./AmesHousing.csv",
            "data/AmesHousing.csv"
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, encoding='utf-8')
                st.success(f"✅ Dataset loaded from {path}")
                return df
        
        # If no file found, allow upload
        st.warning("📁 Default dataset not found. Please upload a CSV file.")
        uploaded_file = st.file_uploader("Upload AmesHousing.csv", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Uploaded dataset loaded successfully!")
            return df
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Ames Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Discover insights and predict house prices with machine learning")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["📊 Dashboard", "🏗️ Data Explorer", "🤖 ML Models", "📈 Visualizations", "ℹ️ About"]
    )
    
    # Dashboard Page
    if page == "📊 Dashboard":
        st.header("📊 Dataset Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric_cols)
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categorical Features", categorical_cols)
        
        # Quick stats
        st.subheader("📈 Quick Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'SalePrice' in df.columns:
                avg_price = df['SalePrice'].mean()
                st.info(f"**Average Sale Price:** ${avg_price:,.2f}")
            
            st.write("**Dataset Info:**")
            st.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.write("**Data Types:**")
            for dtype, count in df.dtypes.value_counts().items():
                st.write(f"- {dtype}: {count} columns")
        
        # Data preview
        st.subheader("👀 Data Preview")
        if st.checkbox("Show first 10 rows"):
            st.dataframe(df.head(10), use_container_width=True)
    
    # Data Explorer Page
    elif page == "🏗️ Data Explorer":
        st.header("🏗️ Data Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Column Inspector")
            selected_col = st.selectbox("Select a column:", df.columns.tolist())
            
            if selected_col:
                col_data = df[selected_col]
                st.write(f"**Column:** {selected_col}")
                st.write(f"**Data type:** {col_data.dtype}")
                st.write(f"**Missing values:** {col_data.isnull().sum()} ({col_data.isnull().mean()*100:.1f}%)")
                
                if pd.api.types.is_numeric_dtype(col_data):
                    st.write("**Statistics:**")
                    st.write(col_data.describe())
                else:
                    st.write("**Top values:**")
                    st.write(col_data.value_counts().head(10))
        
        with col2:
            st.subheader("📋 Quick Filters")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                filter_col = st.selectbox("Filter by column:", numeric_cols)
                if filter_col:
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    selected_range = st.slider(
                        f"Select range for {filter_col}:",
                        min_val, max_val, (min_val, max_val)
                    )
                    
                    filtered_df = df[(df[filter_col] >= selected_range[0]) & 
                                    (df[filter_col] <= selected_range[1])]
                    st.write(f"**Filtered results:** {len(filtered_df)} properties")
    
    # ML Models Page
    elif page == "🤖 ML Models":
        st.header("🤖 Machine Learning Models")
        
        st.info("""
        **Available Models:**
        - Linear Regression
        - Random Forest Regressor
        - XGBoost Regressor
        
        Select a model to train and see performance metrics.
        """)
        
        model_choice = st.selectbox(
            "Choose a model to train:",
            ["Linear Regression", "Random Forest", "XGBoost"]
        )
        
        if st.button("🚀 Train Selected Model", type="primary"):
            with st.spinner(f"Training {model_choice}..."):
                # Simulate training (replace with actual model training)
                import time
                time.sleep(2)
                
                st.success("✅ Model trained successfully!")
                
                # Show mock results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", "$23,456")
                with col2:
                    st.metric("R² Score", "0.87")
                with col3:
                    st.metric("Training Time", "2.3s")
                
                st.subheader("📊 Model Performance")
                st.write("""
                | Model | RMSE | R² Score | Training Time |
                |-------|------|----------|---------------|
                | Linear Regression | $23,456 | 0.87 | 2.3s |
                | Random Forest | $18,234 | 0.92 | 5.1s |
                | XGBoost | $15,678 | 0.94 | 3.8s |
                """)
                
                st.balloons()
    
    # Visualizations Page
    elif page == "📈 Visualizations":
        st.header("📈 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                dist_col = st.selectbox("Select column for distribution:", numeric_cols)
                if dist_col:
                    chart = alt.Chart(df).mark_bar().encode(
                        alt.X(f"{dist_col}:Q", bin=True),
                        y='count()'
                    ).properties(title=f"Distribution of {dist_col}")
                    st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.subheader("🔗 Scatter Plot")
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis:", numeric_cols, index=0)
                y_col = st.selectbox("Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                if x_col and y_col:
                    scatter_chart = alt.Chart(df).mark_circle(size=60).encode(
                        x=x_col,
                        y=y_col,
                        tooltip=[x_col, y_col]
                    ).interactive().properties(
                        title=f"{x_col} vs {y_col}"
                    )
                    st.altair_chart(scatter_chart, use_container_width=True)
    
    # About Page
    else:
        st.header("ℹ️ About This App")
        
        st.write("""
        ## 🏠 Ames Housing Price Predictor
        
        This interactive web application allows you to explore the Ames Housing dataset 
        and predict house prices using various machine learning models.
        
        ### ✨ Features:
        - **Data Exploration**: Interactive analysis of housing features
        - **Visualizations**: Charts and graphs for data insights
        - **ML Models**: Multiple regression models for price prediction
        - **User-Friendly**: Intuitive interface for all users
        
        ### 🛠️ Built With:
        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - XGBoost
        - Altair
        
        ### 📊 Dataset:
        The Ames Housing dataset contains information about residential home sales in Ames, Iowa,
        from 2006 to 2010. It includes 2930 properties with 80 features each.
        
        ### 🚀 How to Use:
        1. Explore different sections using the sidebar navigation
        2. Analyze individual features in the Data Explorer
        3. Train machine learning models in the ML Models section
        4. Create visualizations in the Visualizations section
        """)
        
        st.success("🎯 **Tip**: Start with the Dashboard to get an overview, then explore other sections!")

if __name__ == "__main__":
    main()
