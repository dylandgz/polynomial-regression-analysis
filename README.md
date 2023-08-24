# Polynomial Regression Analysis
Polynomial regression model analysis and testing using batch gradient descent. The code supports various functionalities including polynomial feature generation, mean squared error calculation, learning curve analysis, and model evaluation.

1. **Polynomial Feature Generation:**
   - The function `polynomialFeatures(X, degree)` generates polynomial and interaction features for a given degree of the polynomial.

2. **Mean Squared Error (MSE) Calculation:**
   - The function `mse(y_true, y_pred)` calculates the mean squared error between two vectors.

3. **Learning Curve Calculation:**
   - The function `learning_curve(model, X, Y, cv, train_sizes, learning_rate, epochs, tol, regularizer, lambda_, **kwargs)` computes training and validation errors for different training sizes using cross-validation.

4. **Linear Regression Model:**
   - The class `Linear_Regression` includes methods `fit`, `predict`, and `__init__` for implementing linear regression using batch gradient descent. It supports regularization (L1 or L2) and early stopping.

5. **Standard Scaler:**
   - The class `StandardScaler` includes methods `fit` and `transform` for standardizing data.

6. **Data Processing:**
   - The code reads data from a CSV file into a Pandas DataFrame.
   - Variable summaries are generated using the `describe()` function.
   - Data shuffling is done using Pandas' `sample` function.
   - Pair plots are generated using Seaborn for visualization.
   - The `partition(X, y, t)` function is used to split data into training and testing sets.

7. **Model Evaluation:**
   - Hyperparameter tuning is performed using k-fold cross-validation to evaluate different combinations of lambda, learning rate, and regularization.
   - Model performance is evaluated on the test data using optimal hyperparameters.
   - Learning curves are generated to visualize the model's performance during training.

8. **Feature Dropping Experiment:**
   - Certain features are dropped from the dataset to observe their impact on model performance.
   - Hyperparameter tuning and learning curve analysis are repeated on the modified dataset.
   - 
