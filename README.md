# L2-Regularized Linear Regression for Age Prediction

This project implements L2-regularized linear regression for age prediction from facial images using gradient descent optimization with mini-batch training.

## Overview

The implementation performs age regression on facial images using:
- **L2 Regularization**: Prevents overfitting by penalizing large weights (excluding bias term)
- **Mini-batch Gradient Descent**: Efficient training with configurable batch sizes
- **Hyperparameter Tuning**: Grid search over learning rates, regularization strengths, batch sizes, and epochs
- **Real-time Visualization**: Live plotting of cost function during training

## Features

- **Data Processing**: Loads and preprocesses facial image data (48x48 pixels flattened to 2304 features)
- **Train/Validation Split**: 80/20 split with random permutation
- **Cost Function**: Mean Squared Error with L2 regularization (bias term not regularized)
- **Gradient Computation**: Analytical gradients for efficient optimization
- **Hyperparameter Search**: Comprehensive grid search over multiple parameter combinations
- **Model Selection**: Automatically selects best hyperparameters based on validation cost
- **Results Export**: Saves optimal weights, bias, and all hyperparameter results to files

## Files

- `ques2.py`: Main implementation file
- `age_regression_Xtr.npy`: Training images (48x48 facial images)
- `age_regression_ytr.npy`: Training age labels
- `age_regression_Xte.npy`: Test images
- `age_regression_yte.npy`: Test age labels
- `weights.npy`: Optimal learned weights (saved after training)
- `bias.npy`: Optimal learned bias (saved after training)
- `All weights.csv`: Complete results for all hyperparameter combinations

## Usage

```bash
python ques2.py
```

The script will:
1. Load and preprocess the data
2. Perform grid search over hyperparameters
3. Display real-time training progress
4. Select the best model based on validation performance
5. Evaluate on test set
6. Save optimal weights and results

## Hyperparameters

The implementation searches over:
- **Learning rates (ε)**: [0.0005, 0.001, 0.002, 0.0025]
- **Regularization strength (α)**: [0.1, 0.2, 0.4, 0.8]
- **Mini-batch sizes**: [10, 50, 100, 200]
- **Epochs**: [100, 200, 400, 500]

## Key Functions

- `cost_function(X, y, w, b, alpha)`: Computes MSE with L2 regularization
- `grad_cost_function(X, y, w, b, alpha)`: Computes gradients for weight updates
- `parameter_update(X, y, w, b, epsilon, alpha)`: Updates weights using gradient descent
- `train_regressor()`: Main training function with hyperparameter search
- `dictionary_minimum(dictionary)`: Finds hyperparameters with minimum validation cost

## Dependencies

- numpy
- matplotlib
- csv (built-in)
- math (built-in)
- time (built-in)

## Output

The script outputs:
- Real-time cost function plots during training
- Validation and training costs for each hyperparameter combination
- Best hyperparameter configuration
- Final test set performance
- Saved model weights and bias terms

## Example Output

```
hyperparameters set : Epochs : 100, mini batch size : 200, alpha 0.8, epsilon : 0.002
Answer: [test_cost_value]
```

The implementation automatically selects the hyperparameter combination that achieves the lowest validation cost and reports the corresponding test set performance.