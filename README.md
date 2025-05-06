# Diabetes Prediction using Machine Learning

## Overview
This project focuses on predicting diabetes using a machine learning model trained on the **Diabetes Prediction Dataset** from Kaggle. The project demonstrates a complete data science pipeline, including data exploration, visualization, preprocessing, and model training. A Random Forest Classifier is used to predict diabetes, achieving high accuracy and providing insights into key risk factors.

## Dataset
- **Name**: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Source**: Kaggle
- **Description**: Contains 100,000 records with 9 columns (8 features + 1 target):
  - Features: `age`, `gender`, `bmi`, `hypertension`, `heart_disease`, `smoking_history`, `HbA1c_level`, `blood_glucose_level`
  - Target: `diabetes` (0: No, 1: Yes)
- **Use Case**: Binary classification to predict diabetes based on medical and demographic data.

## Project Steps
1. **Exploratory Data Analysis (EDA)**:
   - Visualized distributions (e.g., diabetes cases, age) using Seaborn and Matplotlib.
   - Analyzed relationships (e.g., BMI vs. blood glucose) with scatter plots.
   - Created a correlation heatmap to identify key features.
2. **Data Preprocessing**:
   - Encoded categorical variables (`gender`, `smoking_history`) using LabelEncoder.
   - Scaled numerical features with StandardScaler.
   - Split data into 80% training and 20% testing sets.
3. **Model Training**:
   - Trained a Random Forest Classifier with 100 estimators.
4. **Model Evaluation**:
   - Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
   - Visualized feature importance to highlight key predictors (e.g., HbA1c_level, blood_glucose_level).
5. **Results**:
   - **Accuracy**: 0.97
   - **Precision**: 0.95
   - **Recall**: 0.69
   - **F1-Score**: 0.80


## Key Findings
- High HbA1c levels and blood glucose levels are strongly correlated with diabetes.
- The Random Forest model achieves 97% accuracy, with strong precision (95%) but moderate recall (69%), indicating room for improvement in detecting positive cases.
- Feature importance analysis highlights `HbA1c_level` and `blood_glucose_level` as top predictors.

## Tools Used
- **Python Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **Environment**: Kaggle Notebook

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jamshedali18/diabetes-prediction.git
   ```
2. **Run on Kaggle**:
   - Open the [Kaggle Notebook](https://www.kaggle.com/your-username/diabetes-prediction-notebook).
   - Add the [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) to your notebook.
   - Copy the code from `Diabetes_Prediction.ipynb` and run all cells.
3. **Local Setup** (Optional):
   - Install dependencies:
     ```bash
     pip install pandas numpy scikit-learn seaborn matplotlib
     ```
   - Download the dataset CSV and update the file path in the notebook.
   - Run the Jupyter Notebook locally:
     ```bash
     jupyter notebook Diabetes_Prediction.ipynb
     ```

## How to Use
- Run the Kaggle notebook to reproduce the results.
- Explore the visualizations to understand data patterns.
- Use the trained model to predict diabetes for new data (after preprocessing).

## Future Improvements
- Experiment with other models (e.g., XGBoost, Logistic Regression).
- Address class imbalance to improve recall.
- Implement cross-validation for robust evaluation.

## License
This project is licensed under the MIT License.


---

*This project was developed to showcase skills in data analysis, visualization, and machine learning for healthcare applications.*
