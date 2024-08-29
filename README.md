# Walmart-Sales-Forecasting
## **1. Project Overview**

**Objective:**
The primary goal of this project is to develop predictive models that forecast weekly sales for Walmart stores. Accurate sales forecasting is crucial for effective inventory management, staffing, and planning promotions, especially around holidays and peak shopping seasons. By leveraging historical sales data, the project aims to predict future sales, taking into account various factors such as holiday effects, seasonal trends, and store characteristics.

**Dataset:**
The dataset used in this project is the Walmart Store Sales Forecasting dataset, which includes information on store sales across multiple departments and weeks, along with additional variables such as holiday indicators and store type.

### **2. Approach**

### **2.1 Data Collection and Exploration**

- **Data Sources:** The dataset includes historical sales data for 45 Walmart stores, spanning various weeks, and is accompanied by metadata such as whether the week contains a major holiday.
- **Exploratory Data Analysis (EDA):**
    - **Distribution Analysis:** Analyzed the distribution of sales to understand typical sales ranges, identifying any skewness in the data.
    - **Correlation Matrix:** Created a correlation matrix to identify relationships between different features, focusing on how various factors might influence sales.
    - **Holiday Impact:** Analyzed sales patterns around holidays to observe any spikes or dips, considering holidays like Super Bowl, Labor Day, Thanksgiving, and Christmas.

### **2.2 Data Preprocessing**

- **Handling Missing Data:**
    - **Imputation:** For missing values, used techniques like mean/median imputation or forward-fill/backward-fill based on the nature of the feature (continuous or time series).
- **Outlier Detection and Removal:**
    - **Z-Score Method:** Calculated Z-scores to identify outliers as data points with a Z-score greater than 3 or less than -3.
    - **IQR Method:** Used the Interquartile Range (IQR) to detect outliers, removing data points that lie below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
- **Data Normalization:**
    - **Min-Max Scaling:** Scaled features like sales figures to a range between 0 and 1 to ensure consistent scaling, which is crucial for models like Ridge and Lasso that are sensitive to the magnitude of feature values.
- **Categorical Encoding:**
    - **One-Hot Encoding:** Applied one-hot encoding to categorical variables such as `Store Type` and `Holiday Type` to convert them into numerical representations suitable for machine learning models.
    - **Label Encoding:** Used label encoding for ordinal features, like days of the week, where the order of categories matters.

### **2.3 Feature Engineering**

- **Date Features:**
    - **Weekday/Weekend Feature:** Created binary features indicating whether a day is a weekday or weekend.
    - **Holiday Feature:** Developed a binary feature indicating whether the week contains a significant holiday.
- **Sales Aggregation:**
    - **Rolling Averages:** Calculated rolling averages of sales over different window sizes (e.g., 3 weeks, 7 weeks) to capture short-term and medium-term trends.
    - **Lag Features:** Introduced lag features to account for previous weeks' sales data, enabling the model to learn from past performance.
- **Interaction Features:**
    - **Store-Week Interaction:** Created interaction features between `Store Type` and `Week of the Year` to capture seasonality specific to store types.
    - **Holiday-Department Interaction:** Developed interaction terms between `Holiday Indicator` and `Department` to account for varying holiday effects across different departments.

### **2.4 Model Selection and Training**

- **Baseline Model:**
    - **Linear Regression:** Started with a simple linear regression model to establish a baseline performance, using all available features without regularization.
- **Regularized Regression Models:**
    - **Ridge Regression:** Applied Ridge regression, which introduces an L2 penalty to the loss function. This model helps prevent overfitting by penalizing large coefficients, especially useful when dealing with multicollinearity in the data.
    - **Lasso Regression:** Implemented Lasso regression, which adds an L1 penalty to the loss function. Lasso not only helps in regularization but also performs feature selection by shrinking less important feature coefficients to zero.
    - **ElasticNet Regression:** Combined the strengths of both Ridge and Lasso using ElasticNet regression, which incorporates both L1 and L2 penalties. This model is particularly effective when there are multiple correlated features.

### **2.5 Model Evaluation**

- **Performance Metrics:**
    - **Mean Absolute Error (MAE):** Calculated MAE to measure the average magnitude of errors in the predictions, providing a clear indication of model accuracy.
    - **Mean Squared Error (MSE):** Used MSE to penalize larger errors more heavily, as it squares the error term before averaging, offering insight into the variance of the errors.
    - **R-Squared (R²):** Evaluated R² to determine the proportion of variance in the dependent variable (sales) that is predictable from the independent variables.
- **Cross-Validation:**
    - **k-Fold Cross-Validation:** Performed k-fold cross-validation with k=5 to ensure the model’s robustness by training it on different subsets of the data and validating on the remaining portions. This technique helps in understanding the model’s performance stability.

### **2.6 Hyperparameter Tuning**

- **Grid Search:**
    - **Ridge/Lasso Alpha Parameter:** Conducted a grid search over a range of alpha values to find the optimal regularization strength for both Ridge and Lasso models.
    - **ElasticNet L1_ratio:** Tuned the `L1_ratio` parameter in ElasticNet to balance the contribution of L1 and L2 penalties, optimizing the model’s performance.

### **2.7 Final Model Selection and Prediction**

- **Model Comparison:**
    - Compared the performance of all models using cross-validation results and selected the best-performing model based on the lowest MAE and MSE, and the highest R².
- **Final Predictions:**
    - Used the selected model to make final predictions on the test set and evaluated its performance, ensuring that it generalizes well to unseen data.

