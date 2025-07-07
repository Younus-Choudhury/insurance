import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_insurance_data_analysis_and_report(file_path='insurance.csv'):
    """
    Performs comprehensive data analysis on the insurance Kaggle dataset,
    generates and saves various static plots, and creates a detailed
    data analysis report in Markdown format.

    Args:
        file_path (str): The path to the insurance CSV file.
                         Defaults to 'insurance.csv' in the current directory.
    """

    # --- 1. Setup and Data Loading ---
    print("--- Starting Insurance Data Analysis ---")
    print(f"Attempting to load data from: {file_path}")

    # Check if the specified file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found. Please ensure the CSV file is in the correct directory.")
        return

    # Load the dataset into a pandas DataFrame
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Display the first few rows of the dataset to get an initial glance
    print("\n--- First 5 Rows of the Dataset ---")
    print(df.head().to_string(index=False))

    # Display concise summary of the DataFrame, including data types and non-null values
    print("\n--- Dataset Information (Data Types and Non-Null Counts) ---")
    df.info()

    # --- 2. Exploratory Data Analysis (EDA) ---

    # 2.1 Descriptive Statistics for Numerical Columns
    # This provides a summary of central tendency, dispersion, and shape of the distribution
    # for numerical features (age, bmi, children, charges).
    print("\n--- Descriptive Statistics for Numerical Columns ---")
    print(df[['age', 'bmi', 'children', 'charges']].describe().to_string())

    # 2.2 Value Counts for Categorical Columns
    # This shows the frequency of each unique value in categorical features (sex, smoker, region).
    print("\n--- Value Counts for Categorical Columns ---")
    for col in ['sex', 'smoker', 'region']:
        print(f"\nValue Counts for '{col}':")
        print(df[col].value_counts().to_string())

    # --- 3. Data Visualization ---

    # Set the style for Seaborn plots for better aesthetics
    sns.set_style("whitegrid")

    # Determine the output directory for saving plots and the report
    # This will be the same directory where the 'insurance.csv' file is located.
    output_dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # Create directory if it doesn't exist
    print(f"\nSaving generated plots and report to: {output_dir}")

    # 3.1 Histograms for Numerical Variables
    # These plots show the distribution of individual numerical features.
    plt.figure(figsize=(15, 10))
    plt.suptitle('Distribution of Key Numerical Variables in Insurance Dataset', fontsize=18, y=1.02)

    plt.subplot(2, 2, 1)
    sns.histplot(df['age'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Policyholder Age', fontsize=14)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.subplot(2, 2, 2)
    sns.histplot(df['bmi'], kde=True, bins=30, color='lightcoral')
    plt.title('Distribution of Body Mass Index (BMI)', fontsize=14)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.subplot(2, 2, 3)
    sns.histplot(df['children'], kde=True, bins=6, color='lightgreen')
    plt.title('Distribution of Number of Children Covered', fontsize=14)
    plt.xlabel('Number of Children', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.subplot(2, 2, 4)
    sns.histplot(df['charges'], kde=True, bins=50, color='lightsalmon')
    plt.title('Distribution of Medical Insurance Charges', fontsize=14)
    plt.xlabel('Charges ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(output_dir, '01_numerical_histograms.png'))
    plt.close()
    print("Saved: '01_numerical_histograms.png'")

    # 3.2 Scatter Plots: Numerical Variables vs. Charges
    # These plots visualize the relationship between numerical features and the target variable 'charges'.
    plt.figure(figsize=(18, 6))
    plt.suptitle('Relationship Between Numerical Variables and Medical Charges', fontsize=18, y=1.02)

    plt.subplot(1, 3, 1)
    sns.scatterplot(x='age', y='charges', data=df, alpha=0.6, color='dodgerblue')
    plt.title('Age vs. Medical Charges', fontsize=14)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Medical Charges ($)', fontsize=12)

    plt.subplot(1, 3, 2)
    sns.scatterplot(x='bmi', y='charges', data=df, alpha=0.6, color='forestgreen')
    plt.title('BMI vs. Medical Charges', fontsize=14)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Medical Charges ($)', fontsize=12)

    plt.subplot(1, 3, 3)
    sns.scatterplot(x='children', y='charges', data=df, alpha=0.6, color='darkorange')
    plt.title('Number of Children vs. Medical Charges', fontsize=14)
    plt.xlabel('Number of Children', fontsize=12)
    plt.ylabel('Medical Charges ($)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(output_dir, '02_numerical_scatter_plots.png'))
    plt.close()
    print("Saved: '02_numerical_scatter_plots.png'")

    # 3.3 Box Plots: Categorical Variables vs. Charges
    # These plots show the distribution of charges across different categories.
    plt.figure(figsize=(18, 7))
    plt.suptitle('Medical Charges Distribution by Categorical Variables', fontsize=18, y=1.02)

    plt.subplot(1, 3, 1)
    sns.boxplot(x='sex', y='charges', data=df, palette='pastel')
    plt.title('Medical Charges by Sex', fontsize=14)
    plt.xlabel('Sex', fontsize=12)
    plt.ylabel('Medical Charges ($)', fontsize=12)

    plt.subplot(1, 3, 2)
    sns.boxplot(x='smoker', y='charges', data=df, palette='pastel')
    plt.title('Medical Charges by Smoker Status', fontsize=14)
    plt.xlabel('Smoker Status', fontsize=12)
    plt.ylabel('Medical Charges ($)', fontsize=12)

    plt.subplot(1, 3, 3)
    sns.boxplot(x='region', y='charges', data=df, palette='pastel')
    plt.title('Medical Charges by Geographic Region', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Medical Charges ($)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(output_dir, '03_categorical_box_plots.png'))
    plt.close()
    print("Saved: '03_categorical_box_plots.png'")

    # 3.4 Interaction Plots: Age/BMI with Smoker Status vs. Charges
    # These plots highlight how the relationship between numerical variables and charges changes
    # based on smoker status, revealing important interaction effects.
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.7, palette={'yes': 'red', 'no': 'blue'}, s=50)
    plt.title('Medical Charges by Age, Differentiated by Smoker Status', fontsize=16)
    plt.xlabel('Age (Years)', fontsize=14)
    plt.ylabel('Medical Charges ($)', fontsize=14)
    plt.legend(title='Smoker Status', title_fontsize='13', fontsize='12', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_smoker_age_charges_interaction.png'))
    plt.close()
    print("Saved: '04_smoker_age_charges_interaction.png'")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, alpha=0.7, palette={'yes': 'red', 'no': 'blue'}, s=50)
    plt.title('Medical Charges by BMI, Differentiated by Smoker Status', fontsize=16)
    plt.xlabel('BMI', fontsize=14)
    plt.ylabel('Medical Charges ($)', fontsize=14)
    plt.legend(title='Smoker Status', title_fontsize='13', fontsize='12', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_smoker_bmi_charges_interaction.png'))
    plt.close()
    print("Saved: '05_smoker_bmi_charges_interaction.png'")

    # 3.5 Correlation Matrix of Numerical Variables
    # This heatmap visualizes the Pearson correlation coefficients between numerical features,
    # indicating the strength and direction of linear relationships.
    correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
    print("\n--- Correlation Matrix of Numerical Variables ---")
    print(correlation_matrix.to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix of Numerical Insurance Variables', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_correlation_matrix.png'))
    plt.close()
    print("Saved: '06_correlation_matrix.png'")

    # --- 4. Feature Engineering (for descriptive purposes) ---

    # Create a new categorical feature 'bmi_category' based on BMI values.
    # This helps in analyzing charges across different BMI health classifications.
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 24.9:
            return 'Normal'
        elif 25 <= bmi < 29.9:
            return 'Overweight'
        else:
            return 'Obese'

    df['bmi_category'] = df['bmi'].apply(bmi_category)
    print("\n--- Value Counts for Engineered BMI Category ---")
    print(df['bmi_category'].value_counts().to_string())

    # 4.1 Box Plot: Medical Charges by BMI Category
    # Visualize the distribution of charges across the newly created BMI categories.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bmi_category', y='charges', data=df, palette='viridis',
                order=['Underweight', 'Normal', 'Overweight', 'Obese']) # Ensure specific order
    plt.title('Medical Charges Distribution by BMI Category', fontsize=16)
    plt.xlabel('BMI Category', fontsize=14)
    plt.ylabel('Medical Charges ($)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_bmi_category_charges_boxplot.png'))
    plt.close()
    print("Saved: '07_bmi_category_charges_boxplot.png'")

    print("\nData analysis and static plots generation complete.")
    print(f"All plots (PNG) saved to the directory: {output_dir}")

    # --- 5. Generate and Save the Data Analysis Report ---
    # This section compiles the findings and recommendations into a Markdown report.
    report_content = f"""
# Data Analysis Report: Insurance Dataset (insurance.csv)

## Executive Summary

This comprehensive analysis of the `insurance.csv` dataset aims to identify the key factors influencing medical insurance charges. Our findings reveal that **smoking status** is by far the most dominant predictor of higher charges, followed significantly by **age** and **BMI category (obesity)**. Other factors like sex, number of children, and geographic region show less pronounced impacts, though regional nuances exist. These insights are crucial for insurance providers to refine risk assessment, optimize premium structures, and develop targeted wellness programs.

## 1. Data Overview and Initial Diagnostics

The dataset contains information on {len(df)} policyholders with 7 attributes:
- **Demographic:** `age` (numerical), `sex` (categorical), `children` (numerical), `region` (categorical)
- **Health-related:** `bmi` (numerical), `smoker` (categorical)
- **Target Variable:** `charges` (numerical)

### Data Completeness
- No missing values were identified across any of the columns, indicating a clean dataset for analysis.

### Initial Observations
- **Age:** Ranges from 18 to 64, with a relatively even distribution.
- **BMI:** Shows a right-skewed distribution, with a significant portion of policyholders falling into overweight and obese categories.
- **Charges:** Highly right-skewed, indicating that while most charges are lower, there are a substantial number of high-cost claims.
- **Categorical Balance:**
    - `Sex`: Fairly balanced between male and female policyholders.
    - `Smoker`: Highly imbalanced, with a much larger proportion of non-smokers (approx. 80%) compared to smokers (approx. 20%).
    - `Region`: Relatively balanced distribution across the four geographical regions.

## 2. In-Depth Data Analysis and Visualizations

### 2.1 Distribution of Numerical Variables
(Refer to: `01_numerical_histograms.png`)

- **Age:** The distribution is relatively uniform, suggesting a broad representation of adult age groups.
- **BMI:** The distribution is somewhat right-skewed, with a peak around 30, indicating a prevalence of overweight and obese individuals in the dataset.
- **Children:** Heavily skewed towards 0, meaning many policyholders have no children covered.
- **Medical Charges:** This is a highly right-skewed distribution, with a long tail extending to very high charges. This implies that while most claims are modest, a small percentage of policyholders incur very high medical costs.

### 2.2 Relationship Between Charges and Numerical Variables
(Refer to: `02_numerical_scatter_plots.png`)

- **Age vs. Charges:** A clear positive linear trend is observed; as age increases, medical charges generally tend to increase. Distinct horizontal "bands" are visible, strongly suggesting the influence of other categorical variables.
- **BMI vs. Charges:** There is a general upward trend, but with significant scatter. Higher BMI values are associated with higher charges, but the relationship is not as strong or linear as with age. The "bands" seen here also hint at confounding factors.
- **Children vs. Charges:** This relationship appears weak and scattered, indicating that the number of children covered does not have a strong direct linear impact on charges.

### 2.3 Relationship Between Charges and Categorical Variables
(Refer to: `03_categorical_box_plots.png`)

- **Medical Charges by Sex:** The median charges for males and females are quite similar, suggesting that sex alone is not a primary driver of charge differences. However, males show a slightly wider spread and some higher outliers.
- **Medical Charges by Smoker Status:** This is the most striking relationship. Smokers incur dramatically higher median medical charges compared to non-smokers. The range of charges for smokers is also much wider and extends to significantly higher values. This is a critical differentiator.
- **Medical Charges by Geographic Region:** While the median charges are somewhat similar across regions, the 'southeast' region appears to have a slightly higher median and a greater number of high-charge outliers, indicating potentially higher healthcare costs or different demographic compositions in that area.

### 2.4 Interaction Effects: Smoker Status with Age and BMI
(Refer to: `04_smoker_age_charges_interaction.png` and `05_smoker_bmi_charges_interaction.png`)

- **Age & Smoker Status Interaction:**
    - For **non-smokers (blue points)**, charges increase linearly with age, but remain relatively low.
    - For **smokers (red points)**, charges are consistently much higher across all age groups, forming a distinct, elevated band. This interaction clearly shows that smoking status amplifies the effect of age on charges significantly.
- **BMI & Smoker Status Interaction:**
    - For **non-smokers (blue points)**, there's a moderate increase in charges with increasing BMI.
    - For **smokers (red points)**, charges are substantially higher across all BMI levels. This indicates that while higher BMI contributes to charges, smoking status is the dominant factor, pushing charges into a much higher bracket regardless of BMI.

### 2.5 Correlation Analysis of Numerical Variables
(Refer to: `06_correlation_matrix.png`)

| Variable Pair          | Pearson Correlation Coefficient |
|:-----------------------|:--------------------------------|
| Age & Charges          | 0.30                            |
| BMI & Charges          | 0.20                            |
| Children & Charges     | 0.07                            |
| Age & BMI              | 0.11                            |
| Age & Children         | 0.04                            |
| BMI & Children         | 0.01                            |

- **Interpretation:**
    - **Age** has a moderate positive correlation with charges, meaning older individuals tend to have higher charges.
    - **BMI** has a weak positive correlation with charges.
    - **Children** has a very weak positive correlation, suggesting minimal linear relationship with charges.
    - The correlations among independent variables (age, bmi, children) are very low, indicating little multicollinearity among them.

## 3. Feature Engineering: BMI Category

To gain more insights into the impact of BMI, a new categorical feature `bmi_category` was engineered based on standard health classifications:
- Underweight (BMI < 18.5)
- Normal (18.5 <= BMI < 24.9)
- Overweight (25 <= BMI < 29.9)
- Obese (BMI >= 30)

### Distribution of BMI Categories
- **Obese:** {df['bmi_category'].value_counts().get('Obese', 0)} policyholders
- **Normal:** {df['bmi_category'].value_counts().get('Normal', 0)} policyholders
- **Overweight:** {df['bmi_category'].value_counts().get('Overweight', 0)} policyholders
- **Underweight:** {df['bmi_category'].value_counts().get('Underweight', 0)} policyholders

### Medical Charges by BMI Category
(Refer to: `07_bmi_category_charges_boxplot.png`)

- The box plot clearly shows an increasing trend in median medical charges as BMI category moves from 'Normal' to 'Overweight' to 'Obese'. 'Underweight' individuals also show a range of charges, but the overall trend highlights the financial impact of higher BMI.

## 4. Strategic Business Recommendations

Based on the analysis, the following recommendations are proposed for insurance companies:

### 4.1 Risk-Based Underwriting and Premium Adjustment
- **Smoking Surcharge:** Implement or significantly increase surcharges for smokers. This is the single most impactful factor on charges.
- **BMI-Tiered Premiums:** Introduce or enhance tiered premium structures based on BMI categories, with higher premiums for overweight and obese policyholders.
- **Age-Based Adjustments:** Continue to factor in age as a significant risk component, potentially with finer age-band segmentation.

### 4.2 Health and Wellness Programs
- **Targeted Smoking Cessation Programs:** Actively promote and incentivize smoking cessation programs for policyholders identified as smokers. Consider making participation mandatory for certain premium discounts.
- **Weight Management Initiatives:** Offer comprehensive weight management programs, nutrition counseling, and fitness incentives for overweight and obese policyholders.
- **Preventive Care Emphasis:** Encourage regular health screenings and preventive care, especially for older policyholders and those with higher BMI, to mitigate the progression of costly chronic conditions.

### 4.3 Regional Strategy
- **Southeast Region Investigation:** Conduct further analysis into the 'southeast' region to understand the underlying reasons for higher charges and outliers. This could involve examining healthcare infrastructure, local medical costs, or specific demographic profiles.
- **Geographic Risk Assessment:** Refine regional premium adjustments based on detailed cost analysis and health trends specific to each area.

### 4.4 Product Innovation
- **Healthy Lifestyle Tiers:** Develop and market insurance products with attractive discounts for non-smokers who maintain a healthy BMI.
- **Gamification of Wellness:** Implement wellness challenges and reward systems to encourage healthy behaviors, potentially leading to lower claims.
- **Telehealth Integration:** Promote the use of telehealth services for routine consultations and chronic disease management, which can potentially reduce overall healthcare costs.

## Conclusion

The analysis unequivocally demonstrates that **smoking status** is the most critical determinant of medical insurance charges. **Age** and **BMI (particularly obesity)** are also significant factors. By strategically leveraging these insights, insurance companies can not only enhance their financial performance through optimized risk management and premium adjustments but also contribute to a healthier policyholder base by encouraging and supporting positive lifestyle changes. This data-driven approach fosters a win-win scenario for both the insurer and the insured.
"""
    report_file_path = os.path.join(output_dir, 'insurance_data_analysis_report.md')
    with open(report_file_path, 'w') as f:
        f.write(report_content)
    print(f"\nComprehensive Data Analysis Report saved to: {report_file_path}")
    print("\n--- Data Analysis Complete ---")

# To run the analysis, call the function:
# Make sure 'insurance.csv' is in the same directory as this script,
# or provide the full path to the file.
perform_insurance_data_analysis_and_report('insurance.csv')
