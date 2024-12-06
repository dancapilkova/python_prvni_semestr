import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Načtení dat
lifestyle_data = pd.read_csv("student_lifestyle_dataset.csv")
info_data = pd.read_csv("us_dataset_real_names.csv")

print(lifestyle_data)
print(info_data)
# Čištění dat

# A) Odstranění duplicit
lifestyle_data.drop_duplicates(inplace=True)
info_data.drop_duplicates(inplace=True)

# B) Identifikace prázdných a neplatných hodnot
# Chybějící hodnoty užitečné pro kontrolu
missing_values_lifestyle = lifestyle_data.isnull().sum()
missing_values_info = info_data.isnull().sum()

# C) Identifikace odlehlých hodnot
# Pro sloupec GPA použijeme pravidlo IQR
q1 = lifestyle_data['GPA'].quantile(0.25)
q3 = lifestyle_data['GPA'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Zjištění odlehlých hodnot
outliers_gpa = lifestyle_data[(lifestyle_data['GPA'] < lower_bound) | (lifestyle_data['GPA'] > upper_bound)]

# D) Nahrazení odlehlých a prázdných hodnot
# Nahrazení odlehlých a prázdných hodnot GPA průměrem
mean_gpa = lifestyle_data['GPA'].mean()
lifestyle_data['GPA'] = lifestyle_data['GPA'].apply(lambda x: mean_gpa if x < lower_bound or x > upper_bound else x)
lifestyle_data['GPA'].fillna(mean_gpa, inplace=True)

# Nahrazení chybějících hodnot u hodin (počet hodin na činnosti nemá překročit 24 hodin denně)
time_columns = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
                'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']
lifestyle_data[time_columns] = lifestyle_data[time_columns].fillna(0)
total_time = lifestyle_data[time_columns].sum(axis=1)
lifestyle_data.loc[total_time > 24, time_columns] = lifestyle_data[time_columns].apply(
    lambda x: 24 * x / total_time, axis=0
)

# E) Převod sloupců na správný datový typ
lifestyle_data['Stress_Level'] = lifestyle_data['Stress_Level'].astype(str)
info_data['age'] = info_data['age'].astype(int)

# Spojení datových rámců
merged_data = pd.merge(lifestyle_data, info_data, left_on='Student_ID', right_on='id')

# Výpočet statistik

# a) MIN a MAX hodnota věku studenta
min_age = merged_data['age'].min()
max_age = merged_data['age'].max()

# b) Průměrné GPA v daném městě
average_gpa_per_city = merged_data.groupby('city')['GPA'].mean().to_dict()

# c) Vykreslení histogramu hodnot GPA
plt.figure(figsize=(8, 6))
sns.histplot(merged_data['GPA'], kde=True, bins=20, color='blue')
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Frequency")
plt.savefig("gpa_histogram.png")
plt.close()

# d) Vykreslení tepelné mapy pro korelaci mezi spánkem a úrovní stresu
merged_data['Stress_Level_Numeric'] = merged_data['Stress_Level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
sleep_stress_corr = merged_data[['Sleep_Hours_Per_Day', 'Stress_Level_Numeric']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(sleep_stress_corr, annot=True, cmap="coolwarm")
plt.title("Correlation: Sleep Hours vs Stress Level")
plt.savefig("heatmap_sleep_stress.png")
plt.close()

# e) Vykreslení tepelné mapy pro korelaci mezi společenskými aktivitami a úrovní stresu
social_stress_corr = merged_data[['Social_Hours_Per_Day', 'Stress_Level_Numeric']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(social_stress_corr, annot=True, cmap="coolwarm")
plt.title("Correlation: Social Hours vs Stress Level")
plt.savefig("heatmap_social_stress.png")
plt.close()
