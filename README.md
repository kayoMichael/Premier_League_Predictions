# Classification Analysis of Soccer Matches

This project shows a full Neural Network workflow to predict the result of soccer matches using the [Premier League](https://github.com/kayoMichael/premier_league) Library.

It also serves as an example of how to use the [Premier League](https://github.com/kayoMichael/premier_league) Library for Machine Learning.

## Dataset
The Dataset contains Advanced Statistics of 13,125 games of the Top 5 European Leagues + EFL championship dating back to the 2018/2019 Season.

Each Row Contains 91 Advanced Metrics in the form of the past 10 game statistics Aggregated and Weighted with an Exponential Strategy for the Home Team and Away Team respectively.

We squash the two columns `home_goals` and `away goals` and replace it with classification values:

- 2: win for home team
- 1: draw
- 0: loss for home team

## Exponential Weighted Average (EWA)

$\text{EWA} = \frac{\sum_{i=1}^n \left( \alpha^{n-i} \cdot X_i \right)}{\sum_{i=1}^n \alpha^{n-i}}$

### Where:
- $\text{EWA}$ = Exponential Weighted Average  
- $X_i$ = The value of the $i$-th data point  
- $\alpha$ = Decay factor, $0 < \alpha \leq 1$  
- $n$ = Total number of data points

### Normalization:
The weights are normalized to ensure they sum to 1:

$w_i = \frac{\alpha^{n-i}}{\sum_{j=1}^n \alpha^{n-j}}$

Thus, the weighted average can be expressed as:

$\text{EWA} = \sum_{i=1}^n w_i \cdot X_i$
## Dataset Distribution

<img width="1134" alt="スクリーンショット 2025-05-12 午後10 13 05" src="https://github.com/user-attachments/assets/a4b1cae0-ec91-426f-8598-c7d1a6dc9547" />


As observed, there are much more Home wins the Away wins and draws. This is a bias that might potentially create problems for the model, but we leave it since it is also a feature in soccer games. (Home Court Advantage is statistically proven to occur)

## Initial Model
We first use a base model to do a simple pass through to assess the required adjustments. Since there are 3 possible results: win for home team, draw for home team, and loss for home team, we use a Soft Max activation function. To start off the model, we use a standard 3 hidden layer network with decreasing neurons and standard dropouts in each layer. We use a standard Cross-entropy loss which is suitable for multi-class classification problems.

### Initial Neural Network Structure:
```
Input -> 128 -> 64 -> 32 -> 3
```
The shows significant overfitting and the model is not able to learn the relationship between the training and target columns. 

### Initial Results
The initial accuracy for rounding the output predictions to the nearest whole number is around the 46.5%

```
Epoch [10/100], Train Loss: 1.0748, Val Loss: 1.0677
Epoch [20/100], Train Loss: 0.9923, Val Loss: 1.1805
Epoch [30/100], Train Loss: 0.9833, Val Loss: 1.2752
Epoch [40/100], Train Loss: 1.0126, Val Loss: 1.2921
Epoch [50/100], Train Loss: 0.9922, Val Loss: 1.3942
Epoch [60/100], Train Loss: 0.9753, Val Loss: 1.5311
Epoch [70/100], Train Loss: 0.9632, Val Loss: 1.6312
Epoch [80/100], Train Loss: 0.9421, Val Loss: 1.6722
Epoch [90/100], Train Loss: 0.9211, Val Loss: 1.6532
Epoch [100/100], Train Loss: 0.8923, Val Loss: 1.543


Test Accuracy: 0.4431

```

## Regularization
We first use a direct approach to combat overfitting. Since we have 182 features, there definitely exists overlapping features like (xG and xAG) that we can eliminate from the training. Thus, we use Lasso Regularization (L1) to both perform feature selection and decrease overfitting. 

"""
Lasso Regularization Formula
"""

## Result After Regularization
```
Epoch [10/100], Train Loss: 1.0097, Val Loss: 0.9505
Epoch [20/100], Train Loss: 1.0014, Val Loss: 0.9482
Epoch [30/100], Train Loss: 0.9991, Val Loss: 0.9450
Epoch [40/100], Train Loss: 0.9991, Val Loss: 0.9479
Epoch [50/100], Train Loss: 0.9966, Val Loss: 0.9476
Epoch [60/100], Train Loss: 0.9954, Val Loss: 0.9487
Epoch [70/100], Train Loss: 0.9963, Val Loss: 0.9510
Epoch [80/100], Train Loss: 0.9978, Val Loss: 0.9521
Epoch [90/100], Train Loss: 0.9952, Val Loss: 0.9486
Epoch [100/100], Train Loss: 0.9945, Val Loss: 0.9513

Test Accuracy: 0.4883

```
The Accuracy increased by 4.52% to 48.83% but the loss is still plateauing although not increasing. This indicates that although the overfitting characteristics are gone, the model has not learned the relationship between the training and target columns.

### Bias
After more research, it is clear that the model has been selecting the more dominant result, Home wins and cannot accurately predict draws.
```
Accuracy for class 0: 0.3421
Accuracy for class 1: 0.0000
Accuracy for class 2: 0.8730
```

## SMOTE
To attempt to fix the bias, we use SMOTE to equalize the occurence of Win, Loss and Draw in the training data to force the model to look at the statistical patterns instead of guessing the higher occurrence class.

```
Epoch [10/100], Train Loss: 1.0485, Val Loss: 1.0102
Epoch [20/100], Train Loss: 1.0435, Val Loss: 1.0074
Epoch [30/100], Train Loss: 1.0447, Val Loss: 1.0097
Epoch [40/100], Train Loss: 1.0424, Val Loss: 1.0079
Epoch [50/100], Train Loss: 1.0435, Val Loss: 1.0083
Epoch [60/100], Train Loss: 1.0441, Val Loss: 1.0045
Epoch [70/100], Train Loss: 1.0413, Val Loss: 1.0070
Epoch [80/100], Train Loss: 1.0409, Val Loss: 1.0072
Epoch [90/100], Train Loss: 1.0394, Val Loss: 1.0066
Epoch [100/100], Train Loss: 1.0421, Val Loss: 1.0063
Loss plot saved as 'loss_plot_classification.png'
Accuracy for class 0: 0.3437
Accuracy for class 1: 0.3055
Accuracy for class 2: 0.6788

Test Accuracy: 0.4428

```

The Accuracy slightly went down but the model is able to predict all classes of outputs which was the intended results. The validation loss is still plateauing, which is a cause for concern that the model is still not adequately learning.

## Plateauing Validation Loss

We have seen that the validation loss has either plateaued or increased as epoch -> 100. To fix this problem, we employ an aggresived method. 

### L2 Regularization
We first add a small L2 Regularization, (around 0.0005) which helps decrease the val loss slightly and increase accuracy

```
Epoch [10/100], Train Loss: 1.0355, Val Loss: 1.009
Epoch [20/100], Train Loss: 1.0234, Val Loss: 1.006
Epoch [30/100], Train Loss: 1.0227, Val Loss: 1.003
Epoch [40/100], Train Loss: 1.0254, Val Loss: 1.007
Epoch [50/100], Train Loss: 0.999, Val Loss: 0.998
Epoch [60/100], Train Loss: 1.01, Val Loss: 1.001
Epoch [70/100], Train Loss: 1.02, Val Loss: 1.002
Epoch [80/100], Train Loss: 1.00, Val Loss: 0.9978
Epoch [90/100], Train Loss: 0.998, Val Loss: 0.9981
Epoch [100/100], Train Loss: 0.995, Val Loss: 1.001
Loss plot saved as 'loss_plot_classification.png'
Accuracy for class 0: 0.3211
Accuracy for class 1: 0.3323
Accuracy for class 2: 0.6823

Test Accuracy: 0.4452
```

### Batch Normalization and Decrease Learning Rate on Plateau

We see that the to aggressively increase accuracy and solve the plateauing loss problem, we apply batch normalization to reduce sensitivity to weight initialization (We see that there is early plateau which could be attributed to this problem).

We increase the Learning Rate from 0.001 -> 0.005 and then add Reduce Learning rate on Plateau to effectively lower the learning rate when the loss starts to plateau.

```
Epoch [10/100], Train Loss: 1.01, Val Loss: 1.2
Epoch [20/100], Train Loss: 0.99, Val Loss: 0.99
Epoch [30/100], Train Loss: 0.972, Val Loss: 0.978
Epoch [40/100], Train Loss: 0.965, Val Loss: 0.988
Epoch [50/100], Train Loss: 0.945, Val Loss: 0.975
Epoch [60/100], Train Loss: 0.955, Val Loss: 0.982
Epoch [70/100], Train Loss: 0.872, Val Loss: 0.967
Epoch [80/100], Train Loss: 0.854, Val Loss: 0.954
Epoch [90/100], Train Loss: 0.843, Val Loss: 0.944
Epoch [100/100], Train Loss: 0.824, Val Loss: 0.942
Loss plot saved as 'loss_plot_classification.png'
Accuracy for class 0: 0.3829
Accuracy for class 1: 0.4057
Accuracy for class 2: 0.7317

Test Accuracy: 0.5068
```

## High Gradient problem
The Gradient on the first hidden layer is especially high and problematic compared to the other layers. Although Gradient Clipping might be a solution, since the first layer is the only layer with a relatively high gradient, the first hidden layer may not be complex enough to grasp the initial input. Thus we add a larger layer (256 neurons) as the first input layer and make the Neural Network a 4 hidden layer structure.
```
Epoch 10 - Gradient Norms:
fc1.weight: 0.8027
fc2.weight: 0.4986
fc3.weight: 0.2865
output.weight: 0.1397

Epoch 20 - Gradient Norms:
fc1.weight: 0.6873
fc2.weight: 0.5715
fc3.weight: 0.3389
output.weight: 0.2503

Epoch 30 - Gradient Norms:
fc1.weight: 0.6725
fc2.weight: 0.5114
fc3.weight: 0.4057
output.weight: 0.3176

Epoch 40 - Gradient Norms:
fc1.weight: 0.8055
fc2.weight: 0.4921
fc3.weight: 0.2752
output.weight: 0.1420

Epoch 50 - Gradient Norms:
fc1.weight: 0.7567
fc2.weight: 0.4827
fc3.weight: 0.3034
output.weight: 0.1680

Epoch 60 - Gradient Norms:
fc1.weight: 0.8177
fc2.weight: 0.4721
fc3.weight: 0.2656
output.weight: 0.1317

Epoch 70 - Gradient Norms:
fc1.weight: 0.8234
fc2.weight: 0.4647
fc3.weight: 0.2558
output.weight: 0.1882

Epoch 80 - Gradient Norms:
fc1.weight: 0.7535
fc2.weight: 0.4234
fc3.weight: 0.2777
output.weight: 0.2256

Epoch 90 - Gradient Norms:
fc1.weight: 0.7834
fc2.weight: 0.4421
fc3.weight: 0.2902
output.weight: 0.1929

Epoch 100 - Gradient Norms:
fc1.weight: 0.7966
fc2.weight: 0.4310
fc3.weight: 0.2848
output.weight: 0.1332
```

## Result After Adding an Extra Hidden Layer

We see an instant improvement in loss reduction and accuracy meaning that the model was able to grasp the 
```
Epoch [10/100], Train Loss: 1.1175, Val Loss: 1.0192
Epoch [20/100], Train Loss: 1.0597, Val Loss: 0.9983
Epoch [30/100], Train Loss: 1.0131, Val Loss: 0.9972
Epoch [40/100], Train Loss: 0.9467, Val Loss: 0.9834
Epoch [50/100], Train Loss: 0.8941, Val Loss: 0.9819
Epoch [60/100], Train Loss: 0.8608, Val Loss: 0.9765
Epoch [70/100], Train Loss: 0.8362, Val Loss: 0.9674
Epoch [80/100], Train Loss: 0.8373, Val Loss: 0.9432
Epoch [90/100], Train Loss: 0.8351, Val Loss: 0.9392
Epoch [100/100],Train Loss: 0.8377, Val Loss: 0.9293
Epoch 110:      Train Loss: 0.8211, Val Loss: 0.9388
Accuracy for class 0: 0.4324
Accuracy for class 1: 0.4557
Accuracy for class 2: 0.7812
Test Accuracy: 0.5564
```


## Final Model

The Model holds a 55.64% accuracy at predicting the result of a soccer match using the past 10 day exponentially weighted aggregate data. The model has a 43.24% accuracy on predicting away win, 45.57% accuracy at predicting draws and a 78.12% accuracy at predicting home wins. 

Since the Baseline Accuracy (Only Predicting Home Win) is 41.25%, the model performs ~14.3% better than the baseline accuracy. 

## Limitations of Dataset
The variance and unpredictability of soccer matches really did hurt my accuracy and loss. (E.g. More shots does not mean wins) But overall a 14% increased accuracy is a relatively modest improvement. In the future, more relevant features and a more serious feature selection may need to be performed for better results.
