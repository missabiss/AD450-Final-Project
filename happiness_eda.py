import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix 
from scipy.stats import linregress


# Load data
happiness_df = pd.read_csv("world_happiness_report.csv")

# 1. EDA
# 1. Summarize the data
# I used .info() to see all of the columns and data types
print()
print("Happiness Dataset Info: \n")
print(happiness_df.info()) 

# 2. Get basic statistics
# I used .describe() to see the ranges of the values and make sure data seems reasonable/real
print()
print("Summary Statistics: \n")
print(happiness_df.describe())

# 3. Get value counts of a categorical column 
# I used .value_counts() to see how the data was spread across countries
print()
print("Value Counts: \n")
for col in happiness_df.columns:
    if happiness_df[col].dtype == "object":
        print(happiness_df[col].value_counts())
        print()

# # 4. Get histograms of numeric columns
# # I used .hist() to see how each column's data is spread
# happiness_df.hist(bins=20, figsize=(12, 10), edgecolor='black')
# plt.tight_layout()
# plt.show()

# # 5. Look for relationships in the data
# # I used scatter_matrix to see if there were any obvious connections between happiness_score and any other variables,
# # or between subsets of variables (all on one scatter_matrix was too difficult to see due to large number of variables)
# features_1 = ['Happiness_Score', 'GDP_per_Capita', 'Social_Support', 'Healthy_Life_Expectancy']
# scatter_matrix(happiness_df[features_1])
# plt.show()
# features_2 = ['Happiness_Score', 'Freedom', 'Generosity', 'Corruption_Perception']
# scatter_matrix(happiness_df[features_2])
# plt.show()
# features_3 = ['Happiness_Score', 'Unemployment_Rate', 'Education_Index', 'Population']
# scatter_matrix(happiness_df[features_3])
# plt.show()
# features_4 = ['Happiness_Score', 'Urbanization_Rate', 'Life_Satisfaction', 'Public_Trust']
# scatter_matrix(happiness_df[features_4])
# plt.show()
# features_5 = ['Happiness_Score', 'Mental_Health_Index', 'Income_Inequality', 'Public_Health_Expenditure']
# scatter_matrix(happiness_df[features_5])
# plt.show()
# features_6 = ['Happiness_Score', 'Climate_Index', 'Work_Life_Balance', 'Internet_Access']
# scatter_matrix(happiness_df[features_6])
# plt.show()
# features_7 = ['Happiness_Score', 'Crime_Rate', 'Political_Stability', 'Employment_Rate']
# scatter_matrix(happiness_df[features_7])
# plt.show()

# 2. Data Cleaning and Transformation
# 1. Filling NaN values
# I used .isna() to find missing values (there were none)
nan_rows = happiness_df[happiness_df.isna().any(axis=1)]
if nan_rows.empty:
    print("There is no missing data.")
else:
    print("Rows with missing data:")
    print(nan_rows)

# 2. Correct data dtype issues
# I used .to_datetime to convert year in int64 to year in datetime
happiness_df['Year'] = pd.to_datetime(happiness_df['Year'], format='%Y')
print("New dtype for Year column: ", happiness_df['Year'].dtype)

# 3. Data Joining
# 1. Merging two or more dataframes on a column
# I created a dataframe of continent, and merged on country
continent_data = {
    'Country': ['USA', 'France', 'Germany', 'Brazil', 'Australia', 'India', 'UK', 'Canada', 'South Africa', 'China'],
    'Continent': ['North America', 'Europe', 'Europe', 'South America', 'Australia', 'Asia', 'Europe', 'North America', 'Africa', 'Asia']
}
continent_df = pd.DataFrame(continent_data)
print(continent_df)
happiness_with_cont = happiness_df.merge(continent_df, on='Country', how='left')
pd.set_option('display.max_columns', None)  # Show all columns
print(happiness_with_cont.head(10))

# 4. Data Visualization
# Has the USA happiness score decreased between 2005 and 2024?
# Plot happiness score of usa vs year
usa_data = happiness_with_cont[happiness_with_cont['Country'] == 'USA'].sort_values('Year')
plt.figure(figsize=(10, 5))
sns.lineplot(data=usa_data, x='Year', y='Happiness_Score', marker='o')
plt.title('Happiness Score Over Time – USA')
plt.xlabel('Year')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.tight_layout()
plt.show()
# Fit a linear regression to see if there is any correlation between year and happiness_score for the USA
usa_data['Year_num'] = usa_data['Year'].dt.year
slope, intercept, r_value, p_value, std_err = linregress(usa_data['Year_num'], usa_data['Happiness_Score'])
print("Happiness Score vs Year Linear Regression Fit: ")
print(f"Slope: {slope:.3f}")
print(f"R-squared: {r_value**2:.3f}")
print(f"P-value: {p_value:.3f}")
# No correlation observed

# What factors correlate most strongly with happiness_score for USA?
# Find the pearson correlation coefficients for each numeric column
# Filter for USA
usa_df = happiness_with_cont[happiness_with_cont['Country'] == 'USA'].copy()
# Select only numeric columns
numeric_usa = usa_df.select_dtypes(include='number')  # this grabs only numeric columns (e.g., float64, int64)
# Compute correlation with Happiness_Score
correlations = numeric_usa.corr()
# Extract correlations with Happiness_Score
happiness_corr = correlations['Happiness_Score'].drop('Happiness_Score')  # drop self-correlation
# Sort by strength of correlation
happiness_corr_sorted = happiness_corr.sort_values(key=abs, ascending=False)
# Print or display
print(happiness_corr_sorted)
# Make bar plot to show these results
plt.figure(figsize=(10, 6))
happiness_corr_sorted.plot(kind='bar')
plt.title('Correlation with Happiness Score (USA)')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()

# Is the correlation between Employment_Rate and Happiness statistically significant?
# Plot happiness score of usa vs Employment_Rate
plt.figure(figsize=(10, 5))
sns.regplot(data=usa_data, x='Employment_Rate', y='Happiness_Score', scatter=True, marker='o', line_kws={"color": "red"})
plt.title('Happiness Score Vs Employment Rate – USA')
plt.xlabel('Employment Rate')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.tight_layout()
plt.show()
# Fit a linear regression to see if there is any correlation between Employment_Rate and Happiness_Score for the USA
slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = linregress(usa_data['Employment_Rate'], usa_data['Happiness_Score'])
print("Happiness Score vs Employment Rate Linear Regression Fit: ")
print(f"Slope: {slope_2:.3f}")
print(f"R-squared: {r_value_2**2:.3f}")
print(f"P-value: {p_value_2:.3f}")
# No correlation observed

# What might be causing the downward trend in Happiness_Score since 2021?
# Filter the data to be just for the USA from 2021-2024
happiness_df['Year_num'] = happiness_df['Year'].dt.year
usa_recent = happiness_df[(happiness_df['Country'] == 'USA') & (happiness_df['Year_num'].between(2021, 2024))]
# Re-evaluate pearson correlations
# Select only numeric columns
numeric_usa_recent = usa_recent.select_dtypes(include='number')  # this grabs only numeric columns (e.g., float64, int64)
# Compute correlation with Happiness_Score
correlations_recent = numeric_usa_recent.corr()
# Extract correlations with Happiness_Score
happiness_corr_recent = correlations_recent['Happiness_Score'].drop('Happiness_Score')  # drop self-correlation
# Sort by strength of correlation
happiness_corr_recent_sorted = happiness_corr_recent.sort_values(key=abs, ascending=False)
# Print or display
print(happiness_corr_recent_sorted)
# Make bar plot to show these results
plt.figure(figsize=(10, 6))
happiness_corr_recent_sorted.plot(kind='bar')
plt.title('Recent Correlation with Happiness Score (USA)')
plt.ylabel('Recent Correlation Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()

# Is the correlation of Happiness Score with healthy life expectancy from 2021-2024 for the USA statistically significant?
# Plot happiness score of usa from 2021-2024 vs Healthy_Life_Expectancy
plt.figure(figsize=(10, 5))
sns.regplot(data=usa_recent, x='Healthy_Life_Expectancy', y='Happiness_Score', scatter=True, marker='o', line_kws={"color": "red"})
plt.title('Happiness Score Vs Healthy Life Expectancy – USA 2021-2024')
plt.xlabel('Healthy Life Expectancy')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.tight_layout()
plt.show()
# Fit a linear regression to see if there is any correlation between 
# Healthy Life Expectancy and Happiness_Score for the USA between 2021-2024?
slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = linregress(usa_recent['Healthy_Life_Expectancy'], usa_recent['Happiness_Score'])
print("Happiness Score vs Healthy Life Expectancy- USA 2021-2024 Linear Regression Fit: ")
print(f"Slope: {slope_3:.3f}")
print(f"R-squared: {r_value_3**2:.3f}")
print(f"P-value: {p_value_3:.3f}")
# No correlation observed

# 5. Aggregation and Grouping
# 1. Perform an aggregation on all data
# Group by Country and Year, then calculate the mean for all numeric columns
aggregated_df = happiness_df.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
