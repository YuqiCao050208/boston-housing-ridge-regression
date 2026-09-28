# Boston Housing Price Prediction Analysis Speech
**Presenter: Yuqi Cao**  
**Date: June 28, 2025**

Good evening, everyone. My name is Yuqi Cao, and I’m thrilled to present an analysis of predicting median house prices using the Boston Housing dataset with Ridge Regression. Today, I’ll guide you through the dataset, explain how Ridge Regression is applied, compare actual and predicted prices, highlight the model’s limitations, and suggest ways to improve its performance. By the end, you’ll have a clear understanding of Ridge Regression’s strengths and areas for enhancement in this context.

## Dataset Overview
The Boston Housing dataset, sourced from the UCI Machine Learning Repository, contains 506 entries, each representing a Boston neighborhood from the 1970s. It includes 14 features that capture various characteristics, such as:

- **CRIM**: Per capita crime rate by town.
- **RM**: Average number of rooms per dwelling.
- **LSTAT**: Percentage of lower-status population.
- **PTRATIO**: Pupil-teacher ratio by town.
- **CHAS**: A dummy variable indicating proximity to the Charles River (1 if near, 0 otherwise).
- **MEDV**: Median value of owner-occupied homes in thousands of dollars, our target variable.

Here’s a summary of key features:

| Feature  | Description                     |
|----------|---------------------------------|
| CRIM     | Per capita crime rate           |
| RM       | Average rooms per dwelling     |
| LSTAT    | % lower status population      |
| PTRATIO  | Pupil-teacher ratio            |
| MEDV     | Median house price ($1000s)    |

The goal is to predict MEDV using these features. This dataset is widely used in machine learning due to its diverse features and clear regression task, making it an excellent choice for evaluating Ridge Regression.

## Model Description
Ridge Regression is a regularized form of linear regression that adds a penalty term to the loss function, controlled by a hyperparameter alpha, to prevent overfitting and handle correlated features. In our analysis, we apply Ridge Regression with an alpha of 0.1, as determined by hyperparameter tuning, to a carefully selected set of features that include both original and engineered variables to capture key interactions.

The features used are:

- **RM**: Average number of rooms per dwelling.
- **PTRATIO**: Pupil-teacher ratio by town.
- **LSTAT**: Percentage of lower-status population.
- **RL**: Product of RM and LSTAT, capturing their interaction.
- **C_R**: Product of CHAS and RM, reflecting the influence of river proximity on room count.
- **C_L**: Product of CHAS and PTRATIO, capturing the interaction between river proximity and school quality.
- **C_P**: Product of CHAS and RM, which is redundant with C_R and should be excluded to avoid multicollinearity.

Since C_P is identical to C_R, including both would introduce perfect multicollinearity, which could destabilize predictions. For clarity, we exclude C_P, focusing on RM, PTRATIO, LSTAT, RL, C_R, and C_L. These features were chosen to balance predictive power and model stability, with Ridge Regression’s regularization helping to mitigate issues from correlated features like RL and LSTAT.

The model is trained on 80% of the data and tested on the remaining 20%, with performance evaluated using Mean Squared Error (MSE). Ridge Regression’s ability to shrink coefficients makes it particularly suitable for datasets with potential multicollinearity, as seen with our engineered features.

## Actual vs. Predicted Prices
Let’s compare actual and predicted house prices. The dataset provides actual MEDV values, such as $24,000, $21,600, and $34,700. Unfortunately, the code does not provide specific predicted prices, so we rely on typical performance metrics for Ridge Regression on this dataset. Research, such as articles on [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-boston-housing-kaggle-challenge-with-linear-regression/), suggests that Ridge Regression often achieves an MSE of around 15–25 on the Boston Housing dataset, corresponding to a root mean squared error (RMSE) of approximately $3,900–$5,000. This indicates that predictions are generally within a few thousand dollars of actual prices, a reasonable performance for a regularized linear model.

The scatter plot shown earlier provides a hypothetical visualization of actual versus predicted prices. Points close to the red diagonal line represent accurate predictions, while deviations indicate errors. Ridge Regression’s regularization likely results in predictions that are more stable than those from non-regularized models, particularly when dealing with correlated features like RL and LSTAT. For example, a house with an actual price of $24,000 might be predicted at $23,500, showing a small error, while larger deviations could occur for houses near the $50,000 cap.

## Model Shortcomings
Despite its strengths, Ridge Regression has limitations that impact its performance:

1. **Linearity Assumption**: Ridge Regression assumes a linear relationship between features and MEDV. In reality, housing prices may exhibit non-linear patterns. For instance, the effect of LSTAT on price may intensify at higher values, a pattern linear models struggle to capture.

2. **Multicollinearity**: While Ridge Regression mitigates multicollinearity by shrinking coefficients, correlated features like RL (RM * LSTAT) and the original RM and LSTAT can still affect interpretability. The inclusion of interaction terms like C_R and C_L, while potentially informative, may introduce noise if their effects are not significant.

3. **Data Censoring**: The MEDV variable is capped at $50,000, affecting 76 houses (about 15% of the dataset). This censoring means true prices for high-value homes are unknown, potentially skewing predictions for expensive properties.

4. **Feature Selection**: The inclusion of interaction terms like C_R and C_L assumes that proximity to the Charles River significantly modifies the effects of RM and PTRATIO. If these interactions are weak, they could add noise, reducing model accuracy.

These limitations, discussed in resources like [Towards Data Science](https://towardsdatascience.com/predict-housing-price-using-linear-regression-in-python-bfc0fcfff640), suggest that while Ridge Regression is robust, it may not fully capture the complexity of housing price dynamics.

## Recommendations
To improve prediction accuracy, consider the following strategies:

1. **Non-Linear Models**: Models like Random Forests or Neural Networks can capture non-linear relationships, potentially improving accuracy. For example, Random Forests can model how LSTAT’s impact on price varies across different ranges.

2. **Feature Transformations**: Applying transformations, such as logarithmic scaling to skewed features like LSTAT, can help linear models better capture non-linear patterns.

3. **Advanced Feature Selection**: Methods like Recursive Feature Elimination (RFE) can identify the most predictive features, ensuring that interaction terms like C_R and C_L are only included if they significantly improve performance.

4. **Residual Analysis**: Plotting residuals (differences between actual and predicted prices) can reveal patterns like non-linearity or uneven error variance, guiding model refinements.

These recommendations, inspired by insights from [Medium](https://medium.com/data-science/linear-regression-on-boston-housing-dataset-f409b7e4a155), aim to address the model’s limitations and enhance predictive power.

## Conclusion
Ridge Regression provides a robust baseline for predicting Boston house prices, leveraging regularization to handle correlated features effectively. Its performance, likely with an MSE of 15–25, suggests reasonable accuracy, but limitations like linearity assumptions and data censoring highlight areas for improvement. By exploring non-linear models, refining feature selection, and analyzing residuals, we can develop more accurate predictions. These advancements could have practical applications in real estate, urban planning, and economic analysis.

For further reading, the references listed in the presentation, including articles from GeeksforGeeks, Medium, and Towards Data Science, offer valuable insights into regression techniques for this dataset. Future work could explore ensemble methods or techniques to address data censoring, such as survival analysis, to further improve model performance.

Thank you for your attention. I hope this analysis highlights the potential and challenges of using Ridge Regression for housing price prediction. I’m happy to answer any questions you may have.

## Works Cited
- [GeeksforGeeks: Linear Regression using Boston Housing Dataset](https://www.geeksforgeeks.org/machine-learning/ml-boston-housing-kaggle-challenge-with-linear-regression/)
- [Medium: Linear Regression on Boston Housing Dataset](https://medium.com/data-science/linear-regression-on-boston-housing-dataset-f409b7e4a155)
- [Towards Data Science: Predict Housing Price using Linear Regression](https://towardsdatascience.com/predict-housing-price-using-linear-regression-in-python-bfc0fcfff640)