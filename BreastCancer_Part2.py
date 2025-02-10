import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
cancer = datasets.load_breast_cancer()

# Display dataset information
print("Feature Names:", cancer.feature_names)
print("Target Names:", cancer.target_names)
print("Size of features:",cancer.data.size)
# Import train_test_split function
#from sklearn.model_selection import train_test_split
X=cancer.data
y=cancer.target
# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y) # 80% training and 20% test
#from sklearn.preprocessing import StandardScaler
# Normalize the dataset for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#from sklearn.ensemble import GradientBoostingClassifier
# Define Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)

# Hyperparameter Tuning with GridSearchCV
parameters = {
    'n_estimators': [50, 100, 200],'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3, 5, 10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]
}
gridSearchCv = GridSearchCV(gb, parameters, cv=5, scoring='accuracy',n_jobs=-1)
gridSearchCv.fit(X_train, y_train)

# Get best model
best_gbm = gridSearchCv.best_estimator_

print("Best Hyperparameters:", gridSearchCv.best_params_)
# k-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_gbm, X_train, y_train, cv=kfold, scoring='accuracy')

print("Cross-Validation Accuracy: {:.4f}".format(np.mean(cv_scores)))
# Train the Best Model and Make Predictions
best_gbm.fit(X_train, y_train)
y_pred = best_gbm.predict(X_test)
# Model Evaluation
acc = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Precision
prec = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1-score
f1 = f1_score(y_test, y_pred)

# Confusion matrix values (TN, FP, FN, TP)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# Sensitivity
sensi = TP / (TP + FN)  

# Specificity
speci = TN / (TN + FP)

# Print all metrics
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Sensitivity: {sensi:.4f}")
print(f"Specificity : {speci:.4f}")