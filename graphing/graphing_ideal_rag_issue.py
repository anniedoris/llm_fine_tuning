import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('graphing/ideal_rag_mistake_analysis.csv')

df_cleaned = df.dropna(subset=['mistake_type'])

# Get unique mistake types
mistake_types = df_cleaned['mistake_type'].unique()

# Create a dictionary to store lists of "Unnamed: 5" values for each mistake type
mistake_values = {mistake_type: df_cleaned[df_cleaned['mistake_type'] == mistake_type]['Unnamed: 5'].tolist() for mistake_type in mistake_types}
print(mistake_values)

# Example data
categories = df_cleaned['mistake_type'].unique().tolist()
values = [
    mistake_values[categories[0]],  # Values for Category 1
    mistake_values[categories[1]],      # Values for Category 2
    mistake_values[categories[2]],  # Values for Category 3
    mistake_values[categories[3]]   # Values for Category 4
]

# Set bar width
bar_width = 0.2

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Calculate the positions of the bars for each category
positions = []
for i in range(len(values)):
    pos = np.arange(len(values[i])) * bar_width + i * (len(max(values, key=len)) + 1) * bar_width
    positions.append(pos)

# Flatten the positions list for setting x-ticks
flattened_positions = [pos for sublist in positions for pos in sublist]

# Flatten the values list for plotting
flattened_values = [val for sublist in values for val in sublist]

# Plot the bars with outlines
for i, pos in enumerate(positions):
    ax.bar(pos, values[i], bar_width, edgecolor='black')

# Setting x-ticks to the center of each category cluster
category_centers = [(positions[i][0] + positions[i][-1]) / 2 for i in range(len(positions))]
ax.set_xticks(category_centers)
ax.set_xticklabels(categories)

# Adding labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('F1 BoW')

# Removing the legend
ax.legend().set_visible(False)

plt.xticks(rotation=10, ha='right')

# Display the plot
plt.savefig('graphing/idealrag_issue.png')


# Example data
sizes = [len(mistake_values[categories[0]]), len(mistake_values[categories[1]]), len(mistake_values[categories[2]]), len(mistake_values[categories[3]])]  # Four different values
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']  # Different colors for each section

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart Example')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the plot
plt.savefig('graphing/idealrag_issue_barchart.png')
