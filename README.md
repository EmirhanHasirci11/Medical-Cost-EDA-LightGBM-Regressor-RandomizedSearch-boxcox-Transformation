# Medical Cost Prediction with EDA and LightGBM Regressor

This project focuses on predicting individual medical costs billed by health insurance. The analysis involves a thorough Exploratory Data Analysis (EDA) to understand the dataset's characteristics, followed by data preprocessing, feature engineering, and the implementation of a LightGBM Regressor. The model's performance is further enhanced through hyperparameter tuning using `RandomizedSearchCV` and normalization of the target variable using the Box-Cox transformation.

## Dataset
[Kaggle](https://www.kaggle.com/code/emirhanhasrc/eda-lightgbm-regressor-randomizedsearch?scriptVersionId=253212068)
The dataset used is the "Insurance" dataset, which contains information about insurance beneficiaries. It consists of 1338 records and 7 attributes.

### Dataset Features

| Feature | Description | Type |
| :--- | :--- | :--- |
| **age** | Age of the primary beneficiary. | Numeric |
| **sex** | Gender of the insurance contractor. | Categorical (male, female) |
| **bmi** | Body Mass Index, providing an understanding of body weight relative to height. | Numeric |
| **children** | Number of children covered by health insurance / Number of dependents. | Numeric |
| **smoker** | Smoking status of the beneficiary. | Categorical (yes, no) |
| **region** | The beneficiary's residential area in the US. | Categorical (southwest, southeast, northwest, northeast) |
| **charges** | **(Target)** Individual medical costs billed by health insurance. | Numeric |

## Exploratory Data Analysis (EDA)

A comprehensive EDA was performed to gain insights into the data and the relationships between features and medical charges.

1.  **Initial Data Inspection**: The dataset was found to be clean with **no missing values**. The data types were appropriate for each column.
2.  **Distributions of Features**:
    *   **Charges (Target Variable)**: A histogram of the `charges` column revealed a strong **right-skewed distribution**, indicating that most beneficiaries have lower costs, while a smaller number have very high costs. This skewness suggests that a transformation of the target variable could be beneficial for the model.
    *   **Other Features**: Histograms for other numerical features showed that `age` has a relatively uniform distribution, while `bmi` is approximately normally distributed.
3.  **Categorical Feature Analysis**:
    *   **Smoker**: A pie chart showed that the vast majority of beneficiaries (**79.52%**) are non-smokers.
    *   **Sex**: The dataset is well-balanced with a nearly equal number of males and females.
    *   **Region**: The distribution across the four regions is also quite balanced.
4.  **Correlation Matrix**: A heatmap was generated to visualize the correlations between numerical features. The most significant findings were:
    *   **`smoker` status has the strongest positive correlation with `charges`**, indicating that smokers tend to have significantly higher medical costs.
    *   `age` and `bmi` also show a moderate positive correlation with `charges`.

## Data Preprocessing and Feature Engineering

The data was prepared for modeling through several key steps:

1.  **Categorical Encoding**:
    *   Binary categorical features (`sex`, `smoker`) were mapped to integers (0 and 1).
    *   The multi-category `region` feature was one-hot encoded using `pd.get_dummies`, with the first category dropped to avoid multicollinearity.
2.  **Data Splitting**: The dataset was split into a training set (75%) and a testing set (25%) to evaluate the model's performance on unseen data.
3.  **Target Variable Transformation**:
    *   Due to the right-skewness of the `charges` variable observed during EDA, a **Box-Cox transformation** was applied to the training set's target variable (`y_train`). This transformation helps to normalize the distribution, which can improve the performance of many regression models.

## Modeling and Evaluation

The LightGBM (Light Gradient Boosting Machine) Regressor was chosen for its high performance and efficiency.

### 1. Baseline LightGBM Model

A baseline `LGBMRegressor` was trained on the preprocessed training data. The model achieved a strong initial performance on the test set:

-   **R² Score**: **0.8637**
-   **Mean Squared Error (MSE)**: 18,458,085.61

This indicates that the baseline model could explain approximately **86.37%** of the variance in medical charges.

### 2. Hyperparameter Tuning with RandomizedSearchCV

To optimize the model, `RandomizedSearchCV` was used to find the best combination of hyperparameters from a predefined grid. This search aimed to minimize the negative root mean squared error.

**Best Parameters Found:**
-   `subsample`: 0.6
-   `reg_lambda`: 1.0
-   `reg_alpha`: 0.5
-   `num_leaves`: 50
-   `n_estimators`: 100
-   `min_child_samples`: 30
-   `max_depth`: -1
-   `learning_rate`: 0.1
-   `colsample_bytree`: 0.8

### 3. Tuned Model Evaluation

The `LGBMRegressor` was retrained using the best parameters identified by the randomized search. The tuned model showed an improvement in performance:

-   **R² Score**: **0.8776**
-   **Mean Squared Error (MSE)**: 16,446,549.93

## Conclusion

The EDA highlighted that being a smoker is the most significant factor leading to higher medical insurance charges, followed by age and BMI. The skewed nature of the `charges` variable was a key characteristic that needed to be addressed.

By applying a **Box-Cox transformation** to the target variable and using a **LightGBM Regressor**, a strong predictive model was developed. The baseline model already performed well with an R² of **86.37%**. Through hyperparameter tuning with `RandomizedSearchCV`, the model's performance was further improved, reaching an **R² score of 87.76%**.

This project successfully demonstrates a robust pipeline for a regression task, from data exploration and feature engineering to advanced modeling and optimization, resulting in a highly accurate model for predicting medical costs.
