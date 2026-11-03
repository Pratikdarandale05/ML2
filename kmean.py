# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 3. Display dataset information
print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df.isnull().sum())

# 4. Select input features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5. Display scatter plot
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Mall Customers")
plt.show()

# 6. Create K-Means model
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# 7. Train model
model.fit(X)

# 8. Get cluster labels
labels = model.labels_
print("Cluster Labels:", labels)

# 9. Add clusters to dataset
df['Cluster'] = labels
print(df.head())

# 10. Get centroids
centroids = model.cluster_centers_
print("Centroids:\n", centroids)

# 11. Display final cluster graph
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1],
            color='red', marker='X', s=200, label='Centroids')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# 12. Display cluster summary
summary = df.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print(summary)