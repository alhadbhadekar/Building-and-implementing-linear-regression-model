# Linear Regression from Scratch

This project demonstrates how to build and implement a linear regression model from scratch using Python. It covers key concepts like linear regression, gradient descent, and the importance of learning rates in model training.

## Table of Contents

1. [Introduction](#introduction)
2. [Linear Regression Model](#linear-regression-model)
3. [Gradient Descent](#gradient-descent)
4. [Model Implementation](#model-implementation)
5. [Data Processing](#data-processing)
6. [Training the Model](#training-the-model)
7. [Model Evaluation and Visualization](#model-evaluation-and-visualization)
8. [Conclusion](#conclusion)

---

## Introduction

Linear Regression is a fundamental algorithm in machine learning used for predicting continuous values. In this project, we implement a simple linear regression model to predict salary based on work experience. The model uses the formula:

```
Y = wX + b
```

Where:
- `Y` is the dependent variable (Salary).
- `X` is the independent variable (Work Experience).
- `w` is the weight (slope).
- `b` is the bias (intercept).

We use **Gradient Descent** to optimize the weights (`w`) and bias (`b`) by minimizing the error between predicted and actual values.

---

## Linear Regression Model

The model is implemented as a class `Linear_Regression`. The key steps for building this model are:
1. **Initialization**: Define the learning rate and number of iterations for training.
2. **Gradient Descent**: Update the weights and bias iteratively to minimize the loss function.
3. **Prediction**: Make predictions using the optimized weights and bias.

---

## Gradient Descent

Gradient Descent is an optimization algorithm that minimizes the loss function by adjusting model parameters (weights and bias) in the direction of the steepest decrease. The weight and bias update rules are:

```
w = w - α * dw
b = b - α * db
```

Where:
- `α` is the learning rate.
- `dw` and `db` are the gradients of the loss function with respect to `w` and `b`.

---

## Model Implementation

### Linear_Regression Class

```python
class Linear_Regression():

  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  def fit(self, X, Y):
    self.m, self.n = X.shape
    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
    Y_prediction = self.predict(self.X)
    dw = - (2 * np.sum((self.X.T).dot(self.Y - Y_prediction))) / self.m
    db = - 2 * np.sum(self.Y - Y_prediction) / self.m
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db

  def predict(self, X):
    return X.dot(self.w) + self.b
```

---

## Data Processing

1. **Loading the Dataset**: The dataset (`salary_data.csv`) contains the work experience and corresponding salary. 
2. **Data Splitting**: Split the data into features (`X`) and target values (`Y`), then divide it into training and test sets using `train_test_split`.

### Code Example:

```python
salary_data = pd.read_csv('/content/salary_data.csv')
X = salary_data.iloc[:, :-1].values  # Features (work experience)
Y = salary_data.iloc[:, 1].values   # Target (salary)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=2)
```

---

## Training the Model

Once the data is processed, the model is trained on the training data. The learning rate is set to `0.01`, and the number of iterations is `1000`.

```python
model = Linear_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X_train, Y_train)

print('Weight:', model.w)
print('Bias:', model.b)
```

---

## Model Evaluation and Visualization

After training the model, predictions are made on the test set, and the results are visualized. The predicted values are plotted alongside the actual values for comparison.

```python
test_data_prediction = model.predict(X_test)

# Visualizing the predicted values and actual values
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, test_data_prediction, color='blue')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
```

---

## Conclusion

This project provides a hands-on approach to implementing linear regression from scratch using Python. It covers the core concepts of the algorithm, including the importance of the learning rate, gradient descent, and the need for iterative optimization. By visualizing the results, you can see how well the model predicts salary based on work experience.

This implementation is a great foundation for understanding more advanced machine learning algorithms. You can expand this project to include multiple features, regularization, and more sophisticated optimization techniques. 

---

### Requirements

- Python 3.x
- Numpy
- Pandas
- Matplotlib
- Scikit-learn (for data splitting)

---

### License

This project is licensed under the MIT License - see the LICENSE file for details.