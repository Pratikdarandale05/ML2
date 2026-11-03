# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load dataset
df = pd.read_csv("Iris.csv")

print(df.head())
print(df.shape)

# 3. Check missing values
print(df.isnull().sum())

# 4. Select input and output
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Create and train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 8. Predict output
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# 9. Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()

# 11. Classification report
print(classification_report(y_test, y_pred))

# 12. Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower = scaler.transform(new_flower)

print("Predicted Species:", model.predict(new_flower)[0])

# 13. Check different K values
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print("K =", k, "Accuracy =", accuracy_score(y_test, pred) * 100)

# 14. Accuracy vs K graph
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.grid()
plt.show()