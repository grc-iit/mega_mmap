import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    'Category': ['Category A', 'Category B', 'Category C'],
    'Group 1': [20, 35, 30],
    'Group 2': [25, 32, 34]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = pd.melt(df, id_vars='Category', var_name='Group', value_name='Value')
print(df_melted)

# Plotting
sns.barplot(data=df_melted, x='Category', y='Value', hue='Group')

# Adding labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Clustered Bar Chart')

# Show plot
plt.show()