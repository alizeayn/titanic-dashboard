import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from main import load_data, clean_data



# Streamlit page config
st.set_page_config(page_title="Titanic Data Analysis Dashboard", layout="wide")

# Custom visualize function for Streamlit
def visualize_data_streamlit(df):
    if df is None:
        st.error("No data to visualize.")
        return
    
    #1. Histogram of age distribution
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Age Distribution of Passengers')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    # 2. Box plot of Fare by Pclass
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.boxenplot(x='Pclass', y='Fare', data=df, ax=ax2)
    ax2.set_title('Fare by Passenger Class')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Fare')
    st.pyplot(fig2)

    # 3. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig3, ax3 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax3)
    ax3.set_title('Correlation Matrix')
    st.pyplot(fig3)

# Main Streamlit app
def main():
    st.title("Titanic Data Analysis Dashboard")
    st.markdown("Explore factors affecting survival rates using exploratory data analysis (EDA).")

    # Load and clean data
    file_path = './data/Titanic-Dataset.csv'

    df = load_data(file_path)
    if df is None:
        st.error("Failed to load data. Check file path.")
        return
    
    df_clean = clean_data(df)
    if df_clean is None:
        st.error("Failed to clean data.")
        return
    st.success("Data loaded and cleaned successfully!")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        pclass_filter = st.selectbox("Select Passenger Class", options=[0,1,2,3], index=0, format_func=lambda x: 'All' if x == 0 else f'Class{x}')
    with col2:
        sex_filter = st.selectbox("Select Gender", options=['All','male','female'])
    with col3:
        age_min, age_max = st.slider("Select Age Range", min_value=0, max_value=80, value=(0,80))
    
    
    # Apply filters
    filtered_df = df_clean.copy()
    if pclass_filter != 0:
        filtered_df = filtered_df[filtered_df['Pclass'] == pclass_filter]
    if sex_filter != 'All':
        sex_map = {'male':0, 'female':1}
        filtered_df = filtered_df[filtered_df['Sex'] == sex_map[sex_filter]]
    if age_min > 0 or age_max < 80:
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]
    
    
    # Display summary stats
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe())


    # Survial Rate Metric 
    survial_rate = filtered_df['Survived'].mean() * 100
    st.metric("Survial Rate (%)", f"{survial_rate:.1f}%", delta=None)
    st.markdown(f"Based on{len(filtered_df)} filtered passengers.")

    # Display filtered data head
    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(10))

    # Download Button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data (CSV)", csv, "titanic_filtered.csv", "text/csv")
    
    # Visualizations
    st.subheader("Visualizations")
    visualize_data_streamlit(filtered_df)
    
    # Insights section
    st.subheader("Key Insights")
    st.markdown("""
    - **Survival Correlation**: Women (Sex=1) had a higher survival rate (~74%) compared to men (~19%).
    - **Class Impact**: Passengers in 1st class had higher fares and better survival chances.
    - **Age Distribution**: Most passengers were between 20-40 years old.
    """)

if __name__ == "__main__":
    main()

