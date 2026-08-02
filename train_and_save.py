import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load training data
train = pd.read_csv("train.csv")

# Define target and features
target = "Survived"
numeric_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
categorical_features = ["Sex", "Embarked"]

X = train[numeric_features + categorical_features]
y = train[target]

# Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Evaluate models using Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 60)
print("Evaluating Classification Models (5-Fold Cross Validation)")
print("=" * 60)

for name, clf in classifiers.items():
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring="accuracy")
    results[name] = scores.mean()
    print(f"{name:20} | Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("=" * 60)

# Select best model
best_model_name = max(results, key=results.get)
best_score = results[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy = {best_score:.4f}")

# Train the best model on the full training set
best_clf = classifiers[best_model_name]
best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", best_clf)
])
best_pipeline.fit(X, y)

# Save the best model
joblib.dump(best_pipeline, "best_titanic_model.joblib")
print(f"Saved best model to 'best_titanic_model.joblib'")

# Load model and make predictions on test.csv
print("\nLoading test.csv and generating predictions...")
test = pd.read_csv("test.csv")

# Make sure features in test are the same
X_test = test[numeric_features + categorical_features]

# Load saved model to verify it works
loaded_pipeline = joblib.load("best_titanic_model.joblib")
test_predictions = loaded_pipeline.predict(X_test)

# Save predictions
predictions_df = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})
predictions_df.to_csv("predictions.csv", index=False)
print("Saved predictions to 'predictions.csv'")
