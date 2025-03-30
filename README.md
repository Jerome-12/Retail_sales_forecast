# Retail Sales Forecast

The aim of this project is to develop a predictive model that forecasts department-wide sales for each store over the next year. Additionally, the project analyzes the effects of markdowns during holiday weeks and provides recommendations based on derived insights to maximize business impact. The process involves,

### 1. Data Collection:
   Datasets are sourced from CSV files, including historical weekly sales, store features, and store types.

### 2. Data Cleaning:
   Using Python and Pandas, the data is cleaned by removing unnecessary columns and handling any missing values.

### 3. Preprocessing the Data:
   Data is merged into a single DataFrame, transforming categorical variables into numeric for analysis.

### 4. Feature Engineering:
   New features are created, such as lagged weekly sales, weekly aggregations, and holiday indicators, to improve the predictive power of the model.

### 5. Training and Testing the Data:
   The dataset is split into training and testing sets. Various regression models (OLS, Random Forest, Gradient Boosting, and Decision Trees) are trained to predict weekly sales.

### 6. Model Evaluation:
   Each model is assessed using metrics such as Mean Squared Error (MSE) and R-squared (R²) to select the best performing model for sales predictions.

### 7. Predictions:
   The selected model is used to predict department-wide sales for each store over the upcoming year.

### 8. Markdown Analysis During Holidays:
   A separate analysis is performed to model the effects of markdowns during holiday weeks, allowing us to evaluate the impact on sales.

### 9. Recommended Actions:
   An assessment of significant predictors from the regression analysis results in strategic recommendations to drive sales based on insights drawn from the data.

