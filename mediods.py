# 1. Install library (run once)
# !pip install scikit-learn-extra

# 2. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

# 3. Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 4. Display dataset information
print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df.isnull().sum())

# 5. Select input features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print(X.head())

# 6. Display scatter plot
plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Mall Customers")
plt.show()

# 7. Create K-Medoids model
model = KMedoids(n_clusters=5, random_state=42, method='alternate')

# 8. Train model
model.fit(X)

# 9. Get cluster labels
labels = model.labels_
print("Cluster Labels:", labels)

# 10. Add clusters to dataset
df['Cluster'] = labels
print(df.head())

# 11. Get medoids
medoids = model.cluster_centers_
print("Medoids:\n", medoids)

# 12. Display final cluster graph
plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=labels, cmap='viridis', s=60)

plt.scatter(medoids[:, 0], medoids[:, 1],
            color='red', marker='X', s=200, label='Medoids')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Medoids")
plt.legend()
plt.show()

# 13. Display cluster summary
summary = df.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print(summary)