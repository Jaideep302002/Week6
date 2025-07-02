Titanic Survival Prediction: Model Evaluation and Hyperparameter Tuning

This project demonstrates how to build and evaluate machine learning models for predicting Titanic passenger survival using the Titanic dataset. The pipeline includes data preprocessing, training multiple models, evaluating their performance, and optimizing hyperparameters using `GridSearchCV`.

# Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) or [GitHub Link Used](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)
- Features used:
  - `Pclass`
  - `Sex`
  - `Age`
  - `SibSp`
  - `Parch`
  - `Fare`
  - `Embarked`

Target variable: `Survived` (1 = survived, 0 = did not survive)

## Steps Performed

### Data Preprocessing
- Handled missing values (`Age`, `Embarked`)
- One-hot encoded categorical features (`Sex`, `Embarked`)
- Scaled numerical features using `StandardScaler`

### Model Training
Trained and evaluated the following models:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Model Evaluation Metrics
Used the following metrics to compare model performance:
- Accuracy
- Precision
- Recall
- F1 Score

### Hyperparameter Tuning
Used `GridSearchCV` for Random Forest to find the best parameters:
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}
