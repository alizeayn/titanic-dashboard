# Titanic Data Analysis Dashboard

## Overview
This project is a straightforward exploratory data analysis (EDA) dashboard for the Titanic dataset. It focuses on loading, cleaning, and visualizing the data to spot patterns in survival rates, such as the impact of gender, class, and age. No ML models yet—just solid data wrangling and interactive exploration using Pandas and Streamlit.

## Features
- Data loading and cleaning: Handles missing values (e.g., Age filled with mean) and encodes categoricals like Sex.
- Interactive filters: By passenger class, gender, and age range—updates stats and plots in real time.
- Key metrics: Live survival rate calculation based on filters.
- Visualizations: Age histogram, fare boxplot by class, and correlation heatmap.
- Export option: Download filtered data as CSV.

## Setup Instructions
1. Clone the repo:
git clone https://github.com/alizeayn/titanic-dashboard.git
cd titanic-dashboard
text2. Install dependencies:
pip install -r requirements.txt
text3. Add the Titanic dataset (`Titanic-Dataset.csv` from Kaggle) to the `data/` folder.
4. Run the app:
streamlit run app.py
textVisit http://localhost:8501.

## Key Insights
- Women had a ~74% survival rate compared to ~19% for men (correlation: 0.54).
- First-class passengers had higher fares and better odds (~63% survival).
- Age didn't strongly correlate with survival (0.04), but most passengers were 20-40 years old.

## Tech Stack
- Data: Pandas, NumPy
- Viz: Matplotlib, Seaborn
- App: Streamlit

## Files
- `main.py`: Core functions (load, clean, visualize).
- `app.py`: Streamlit dashboard.
- `requirements.txt`: Dependencies.

## License
MIT License.

---
Built by ALi Zeynali | [GitHub](https://github.com/alizeayn)
