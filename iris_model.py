# iris_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  # Decimal (e.g., 0.966)
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Percentage (e.g., 96.67%)

# Save model
#joblib.dump(model, "iris_model1.pkl")


# Save both model and accuracy
joblib.dump({'model': model, 'accuracy': accuracy}, "iris_model.pkl")