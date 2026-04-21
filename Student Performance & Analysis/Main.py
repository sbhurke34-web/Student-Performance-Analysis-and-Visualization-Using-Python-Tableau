# ================== IMPORT LIBRARIES ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== SET OUTPUT FOLDER ==================
output_folder = "visualization"   # already created folder

# ================== LOAD DATA ==================
df = pd.read_csv("student_performance_with_missing_duplicates - student_performance_with_missing_duplicates.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nData Types:\n", df.dtypes)

# ================== DATA CLEANING ==================

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill numerical columns with mean
num_cols = ['math_score', 'science_score', 'study_hours_per_day', 'attendance_percentage']
for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical columns with mode
cat_cols = ['gender', 'parent_education', 'result']
for col in cat_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Remove duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape After Removing Duplicates:", df.shape)

# ================== OUTLIER DETECTION ==================
def detect_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]

outliers_math = detect_outliers('math_score')
print("\nMath Score Outliers:\n", outliers_math)

# ================== DESCRIPTIVE STATISTICS ==================
print("\nMath Score Stats:\n", df['math_score'].describe())
print("\nScience Score Stats:\n", df['science_score'].describe())

# ================== GROUPBY ANALYSIS ==================
print("\nGender vs Math Score:\n",
      df.groupby('gender')['math_score'].agg(['count', 'mean', 'max']))

print("\nResult vs Attendance:\n",
      df.groupby('result')['attendance_percentage'].agg(['mean', 'max']))

# ================== VISUALIZATION (SAVE ALL) ==================

# 1. Histogram
plt.figure()
plt.hist(df['math_score'], bins=10)
plt.title("Distribution of Math Scores")
plt.savefig(f"{output_folder}/hist_math_score.png")
plt.close()

# 2. Boxplot
plt.figure()
plt.boxplot(df['math_score'])
plt.title("Boxplot - Math Score")
plt.savefig(f"{output_folder}/boxplot_math_score.png")
plt.close()

# 3. Bar Chart
plt.figure()
df.groupby('gender')['math_score'].mean().plot(kind='bar')
plt.title("Average Math Score by Gender")
plt.savefig(f"{output_folder}/bar_gender_math.png")
plt.close()

# 4. Horizontal Bar
plt.figure()
df.groupby('parent_education')['math_score'].mean().plot(kind='barh')
plt.title("Parent Education vs Math Score")
plt.savefig(f"{output_folder}/barh_parent_education.png")
plt.close()

# 5. Scatter Plot
plt.figure()
plt.scatter(df['study_hours_per_day'], df['math_score'])
plt.xlabel("Study Hours")
plt.ylabel("Math Score")
plt.title("Study Hours vs Math Score")
plt.savefig(f"{output_folder}/scatter_study_vs_math.png")
plt.close()

# 6. Pie Chart
plt.figure()
df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.savefig(f"{output_folder}/pie_gender.png")
plt.close()

# 7. Line Plot
plt.figure()
plt.plot(df['math_score'])
plt.title("Line Plot of Math Scores")
plt.savefig(f"{output_folder}/line_math_score.png")
plt.close()

# 8. Stacked Bar Chart
plt.figure()
pd.crosstab(df['gender'], df['result']).plot(kind='bar', stacked=True)
plt.title("Gender vs Result")
plt.savefig(f"{output_folder}/stacked_gender_result.png")
plt.close()

# 9. Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig(f"{output_folder}/heatmap.png")
plt.close()

# 10. Violin Plot
plt.figure()
sns.violinplot(x='gender', y='math_score', data=df)
plt.title("Violin Plot")
plt.savefig(f"{output_folder}/violin_plot.png")
plt.close()

# ================== SAVE CLEANED DATA ==================
df.to_csv("cleaned_student_data.csv", index=False)

print("\n✅ All graphs saved in 'visualization' folder")
print("✅ Cleaned dataset saved as 'cleaned_student_data.csv'")