# Loss Functions, Gradient Descent, and Backpropagation

## Lesson 1: Understanding Loss Functions

### Common Loss Functions
1. **Mean Squared Error (MSE)**
   - Mathematical formulation:
     The Mean Squared Error is calculated as: \[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]
     where \(y_i\) is the true value and \(\hat{y}_i\) is the predicted value.
   
   - Use cases:
     - Regression problems
     - Continuous value prediction
     - Signal processing
     
   - PyTorch Implementation:
     ```python
     import torch
     import torch.nn as nn

     # Define MSE Loss
     criterion = nn.MSELoss()

     # Example usage
     predictions = torch.tensor([1.0, 2.0, 3.0])
     targets = torch.tensor([0.9, 2.1, 3.2])
     
     loss = criterion(predictions, targets)
     print(f"MSE Loss: {loss.item()}")
     ```

   - Advantages and disadvantages:
     - Advantages:
       * Simple to understand and implement
       * Differentiable everywhere
       * Convex function, guaranteeing a global minimum
     - Disadvantages:
       * Sensitive to outliers
       * May not be suitable for classification tasks
       * Can lead to slower convergence compared to other loss functions

2. **Cross-Entropy Loss**
   - Binary cross-entropy:
     Used for binary classification problems.
     \[ BCE = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)] \]
     
     PyTorch Implementation:
     ```python
     # Binary Cross Entropy
     criterion = nn.BCELoss()
     
     # Example with binary classification
     predictions = torch.tensor([0.7, 0.3, 0.9])
     targets = torch.tensor([1.0, 0.0, 1.0])
     
     loss = criterion(predictions, targets)
     print(f"Binary Cross Entropy Loss: {loss.item()}")
     ```

   - Categorical cross-entropy:
     Used for multi-class classification problems.
     \[ CE = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) \]
     where C is the number of classes.
     
     PyTorch Implementation:
     ```python
     # Cross Entropy for multi-class classification
     criterion = nn.CrossEntropyLoss()
     
     # Example with 3 classes
     predictions = torch.tensor([[0.2, 0.5, 0.3],
                               [0.8, 0.1, 0.1],
                               [0.1, 0.2, 0.7]])
     targets = torch.tensor([1, 0, 2])  # Class indices
     
     loss = criterion(predictions, targets)
     print(f"Categorical Cross Entropy Loss: {loss.item()}")
     ```

   - Properties and characteristics:
     - Measures the difference between probability distributions
     - Output is always non-negative
     - Provides stronger gradients compared to MSE for classification
     - Works well with probability outputs (softmax/sigmoid)

   - Practical considerations:
     - Use BCELoss when output layer uses sigmoid activation
     - Use CrossEntropyLoss when output layer uses softmax activation
     - Handle numerical stability through log-sum-exp trick (built into PyTorch)
     - Ensure proper input normalization

### Loss Function Selection
1. **Criteria**
   - Problem type
   - Data distribution
   - Model architecture
   - Training stability

## Lesson 2: Gradient Descent Optimization

### Basic Gradient Descent
1. **Algorithm**
   - Mathematical foundation
   - Update rule
   - Learning rate
   - Convergence properties
   - [A Beginner's Guide to Gradient Descent in Machine Learning](https://medium.com/@yennhi95zz/4-a-beginners-guide-to-gradient-descent-in-machine-learning-773ba7cd3dfe)


### Mini-batch Processing
1. **Batch Size Selection**
   - Memory constraints
   - Computational efficiency
   - Statistical properties
   - Training dynamics

2. **Implementation**
   - DataLoader configuration
   - GPU utilization
   - Memory management
   - Parallel processing

## Lesson 4: Backpropagation Algorithm Implementation

### Theory

[Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)

1. **Chain Rule** 
   - Mathematical foundation
   - Computational graphs
   - Gradient flow
   - Auto-differentiation

2. **Implementation Steps**
   - Forward pass
   - Backward pass
   - Parameter updates
   - Gradient checking

### Practical Implementation
1. **PyTorch Autograd**
   - Computational graphs
   - Gradient computation
   - Memory management
   - Common pitfalls

2. **Custom Gradients** [Advanced]
   - Manual implementation
   - Custom autograd functions
   - Gradient checking
   - Debugging strategies

## Lesson 5: Advanced Optimization Techniques

### Modern Optimizers
1. **Adam**
   - Algorithm details
   - Hyperparameters
   - Implementation
   - Best practices

2. **RMSprop**
   - Mathematical foundation
   - Implementation details
   - Parameter tuning
   - Use cases

### Advanced Topics
1. **Second-Order Methods**
   - Newton's method
   - Quasi-Newton methods
   - Natural gradient
   - Practical considerations

2. **Optimization Challenges**
   - Saddle points
   - Local minima
   - Plateau regions
   - Gradient noise

### Best Practices
1. **Hyperparameter Tuning**
   - Grid search
   - Random search
   - Bayesian optimization
   - Cross-validation

2. **Monitoring and Debugging**
   - Learning curves
   - Gradient statistics
   - Loss landscapes
   - Debugging strategies 