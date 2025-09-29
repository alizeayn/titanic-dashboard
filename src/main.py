import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os




# Loading dataset data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Shape:{df.shape[0]} rows, {df.shape[1]} columns")    
        print("Head:\n", df.head(10))
        print("Describe:\n", df.describe())
        df.info()
        print("\n Missing values:\n", df.isnull().sum())
        return df
    except FileNotFoundError:
        print("File not found!")
        return None
    

# Clean data
def clean_data(df):
    if df is None:
        return None
    # Fill NaN in Age with rounded mean
    df['Age'] = df['Age'].fillna(np.round(df['Age'].mean()))

    # Fill NaN in Embarked with mode 
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Drop unnecessary columns
    df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

    # Map Sex
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})

    print("Data cleaned successfully!")
    return df



# Visualize data
def visualize_data(df, output_dir='outputs'):
    if df is None:
        return
    
    # Ensure outputs directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histogram of age distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution of Passengers')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.close()

    # 2. Boxplot of fare by Pclass
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Pclass', y='Fare', data=df)
    plt.title('Fare by Passengers Class')
    plt.xlabel('Class')
    plt.ylabel('Fare')
    plt.savefig(os.path.join(output_dir, 'fare_by_class.png'))

    # 3. Correlation heatmap (numeric columns only)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))

    print("Visualizations saved successfully to outputs!")


if __name__ == "__main__":
    df = load_data('./data/Titanic-Dataset.csv')
    df_clean = clean_data(df)
    if df_clean is not None:
        print("\n Cleaned data info:")
        df_clean.info()
        print("\n Missing values after cleaning:\n", df_clean.isnull().sum(), df_clean.head(10))
        visualize_data(df_clean)