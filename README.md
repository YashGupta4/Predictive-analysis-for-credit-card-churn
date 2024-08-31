
# Predictive Analysis of Customer Churn

## Overview

This repository contains a Python-based project for predictive analysis of customer churn. The goal of this project is to identify customers who are likely to leave a service or product, using various machine learning techniques. The project covers the entire pipeline, from data loading and preprocessing to model building, evaluation, and interpretation.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The repository is structured as follows:

```
├── data/                           # Contains the dataset files
├── notebooks/                      # Jupyter notebooks for EDA and other analyses
├── models/                         # Saved machine learning models
├── src/                            # Source code for data processing and model training
│   ├── data_processing.py          # Functions for data preprocessing
│   ├── model_training.py           # Functions for model training and evaluation
│   └── visualization.py            # Functions for creating plots and visualizations
├── tests/                          # Unit tests for the code
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview and instructions
```

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/YashGupta4/Predictive-analysis-for-credit-card-churn.git
cd Predictive-analysis-for-credit-card-churn
pip install -r requirements.txt
```

## Data

The data used for this project is a customer churn dataset, typically structured with features such as customer demographics, account information, and usage metrics. The dataset is stored in the `data/` directory.

**Key Features:**
- `Customer_Age`: The age of the customer.
- `Dependent_count`: The number of dependents.
- `Months_on_book`: The duration for which the customer has been with the service.
- `Total_Relationship_Count`: Total number of products held by the customer.
- `Attrition_Flag`: Target variable indicating whether the customer churned or not.

## Exploratory Data Analysis (EDA)

The EDA phase involves understanding the dataset's structure, distributions, and relationships between features. Key steps include:

- Descriptive statistics for numerical and categorical features.
- Visualization of feature distributions, separated by the churn status.
- Analysis of correlations between features.

## Data Preprocessing

Data preprocessing involves preparing the data for machine learning by:

- Handling missing values.
- Encoding categorical variables.
- Scaling numerical features.
- Balancing the dataset using SMOTE to handle class imbalance.

## Modeling

The following machine learning models were implemented:

- **Logistic Regression**: A baseline model for binary classification.
- **Random Forest Classifier**: An ensemble method that combines multiple decision trees.
- **Gradient Boosting Classifier**: Another ensemble method that focuses on improving the model's weaknesses.

Each model was trained on the processed data and evaluated using metrics such as accuracy, precision, recall, and ROC-AUC.

## Evaluation

Models were evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of true positives among the predicted positives.
- **Recall**: The proportion of true positives among the actual positives.
- **Confusion Matrix**: A summary of prediction results on a classification problem.
- **ROC-AUC Score**: The area under the receiver operating characteristic curve.

## Feature Importance

Feature importance was analyzed using SHAP (SHapley Additive exPlanations) values to interpret the contribution of each feature to the model's predictions.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

For any questions or feedback, feel free to contact me at [info.guptayash@gmail.com](mailto:info.guptayash@gmail.com).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
