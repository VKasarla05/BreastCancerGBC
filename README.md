# BreastCancerGBC
This project applies GBC to diagnose breast cancer using the Breast Cancer Wisconsin (Diagnostic) dataset. GBC is an ensemble learning technique that builds multiple weak learners-decision trees one after another, correcting errors in every step to ultimately come up with a more accurate model.
Breast Cancer Diagnosis using Gradient Boosting Classifier

Dataset:
The dataset contains 569 samples, each representing a breast tumor, with 30 feature variables describing characteristics extracted from digitized images of a breast mass, such as:

- Mean radius
- Mean texture
- Mean concavity
And more...
The target variable is binary:

- 0: Malignant (Cancerous)
- 1: Benign (Non-Cancerous)

Model Implementation:
Data Splitting: The dataset is split into 80% training and 20% testing.


Scaling: Features are scaled using StandardScaler to ensure improved model performance.


Hyperparameter Tuning: GridSearchCV is applied to optimize the hyperparameters:
Learning Rate: 0.2
Max Depth: 3
Min Samples Leaf: 2
Min Samples Split: 2
Number of Estimators: 100
Cross-Validation: 5-fold cross-validation is used to ensure reliable performance evaluation.

Results:
The Gradient Boosting Classifier achieved the following performance metrics:

Accuracy: 95.61%
Precision: 94.67%
Recall (Sensitivity): 98.61%
F1-Score: 96.60%
Specificity: 90.48%

Conclusion:
The Gradient Boosting Classifier model demonstrated excellent performance with a high recall of 98.61%, meaning that almost all cancerous cases were correctly identified. This is crucial in a medical context to minimize false negatives (missed cancer diagnoses). The precision of 94.67% further reduces false positives, ensuring reliable diagnoses. The balanced performance across precision, recall, and specificity makes this model highly suitable for the diagnosis of breast cancer.

Technologies Used:
Python
Scikit-learn
Pandas
Numpy
Jupyter Notebook (optional for interactive analysis)

Instructions:
Clone the repository.
Install the required dependencies using the requirements.txt file.
Run the BreastCancer_Part2.py script to train the Gradient Boosting model and view the evaluation results.
For a detailed analysis, check out the Jupyter notebook BreastCancer_Part2.ipynb.
