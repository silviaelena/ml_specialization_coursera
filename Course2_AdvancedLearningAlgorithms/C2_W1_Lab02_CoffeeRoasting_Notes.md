# Coffee Roasting Lab - Dataset and Implementation Notes

This page contains notes and explanations for the Coffee Roasting neural network lab (C2_W1_Lab02_CoffeeRoasting_TF.ipynb).

---

## Dataset: Understanding X and Y

The coffee roasting dataset is used to demonstrate binary classification with neural networks. The goal is to predict whether a coffee roast will be good or bad based on roasting parameters.

### X (Features/Input)

**Shape**: `(200, 2)` - 200 samples, 2 features

X contains the **coffee roasting parameters**:
- **Column 0** (`X[:,0]`): **Temperature** in Celsius (range: 150-285°C)
- **Column 1** (`X[:,1]`): **Duration** in minutes (range: 11.5-15.5 minutes)

#### Example Data:
```python
X,Y = load_coffee_data()
print(X.shape)  # (200, 2)

# Sample data points:
[[259.92  11.87]  # Temperature: 259.92°C, Duration: 11.87 min
 [231.01  14.41]  # Temperature: 231.01°C, Duration: 14.41 min
 [175.37  11.72]  # Temperature: 175.37°C, Duration: 11.72 min
 ...
]
```

### Y (Labels/Output)

**Shape**: `(200, 1)` - 200 samples, 1 label

Y contains the **roasting quality** (binary classification):
- **1**: Good roast ✓ (plotted as red X's)
- **0**: Bad roast ✗ (plotted as blue circles)

### The "Good Roast" Criteria

According to the `load_coffee_data()` function in `lab_coffee_utils.py`, a coffee roast is labeled as **good** (Y=1) if **all** of the following conditions are met:

```python
if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
    Y[i] = 1  # Good roast
else:
    Y[i] = 0  # Bad roast
```

Where:
- `t` = Temperature (Celsius)
- `d` = Duration (minutes)
- `y = -3/(260-175)*t + 21` (a linear boundary)

#### Conditions for Good Roast:

1. **Temperature range**: 175°C < temperature < 260°C
2. **Duration range**: 12 min < duration < 15 min
3. **Below the boundary line**: duration ≤ (-3/85) × temperature + 21

This creates a **bounded region** in the feature space where good roasts occur.

---

## Data Normalization

### What is Normalization in Machine Learning?

**Important**: Normalization in ML is **NOT** about orthogonal vectors or unit-length vectors from linear algebra. It's about **scaling features to similar ranges**.

### Z-Score Normalization (Standardization)

The lab uses **z-score normalization**, which transforms data using:

**Formula:**
```
x_normalized = (x - μ) / σ
```

Where:
- `μ` (mu) = mean of the feature
- `σ` (sigma) = standard deviation of the feature

### Implementation in TensorFlow

```python
# Step 1: Create normalization layer
norm_l = tf.keras.layers.Normalization(axis=-1)

# Step 2: Learn the mean and std from training data
norm_l.adapt(X)  # Computes μ and σ for each feature

# Step 3: Apply normalization
Xn = norm_l(X)   # Xn = (X - μ) / σ

# Step 4: IMPORTANT - Use same normalization for test data!
X_testn = norm_l(X_test)  # Uses SAME μ and σ learned from training
```

### Before and After Normalization

**Before normalization:**
```python
Temperature: Min=150.00°C, Max=284.99°C    (range: ~135)
Duration:    Min=11.50min, Max=15.48min    (range: ~4)
```

**After normalization:**
```python
Temperature: Min≈-1.5, Max≈1.5    (centered around 0, std ≈ 1)
Duration:    Min≈-1.5, Max≈1.5    (centered around 0, std ≈ 1)
```

### Example Transformation

```python
# Original data
X = [[185.32, 12.69],   # Temperature: 185°C, Duration: 12.69min
     [259.92, 11.87]]   # Temperature: 260°C, Duration: 11.87min

# After normalization (example values)
Xn = [[-0.45, 0.12],    # Both features now on similar scale!
      [ 1.23, -0.87]]
```

### Why Normalize? Three Critical Reasons

#### 1. Equal Feature Importance
Without normalization, the model treats larger-valued features as more important:

```python
# Without normalization:
Temperature: 150-285  (large numbers)
Duration:    11.5-15.5 (small numbers)
# Neural network thinks temperature is 10-20x more important!

# With normalization:
Temperature: -1.5 to 1.5  (similar scale)
Duration:    -1.5 to 1.5  (similar scale)
# Now both features contribute equally
```

#### 2. Faster Training Convergence
Gradient descent converges much faster when features are normalized:
- **Without normalization**: Gradient descent takes zigzag path, slow convergence
- **With normalization**: Direct path to minimum, fast convergence

#### 3. Numerical Stability
- Prevents overflow/underflow in calculations
- Activation functions (sigmoid, ReLU) work better with inputs near 0
- Weights remain in reasonable ranges

### Critical Rule: Consistent Normalization

**Always use the SAME normalization parameters (μ and σ) for training, validation, and test data!**

```python
# CORRECT: Learn from training data, apply to all
norm_l.adapt(X_train)       # Learn μ and σ from training data
X_train_norm = norm_l(X_train)
X_test_norm = norm_l(X_test)  # Use SAME μ and σ

# WRONG: Recomputing for test data
test_norm_l = tf.keras.layers.Normalization()
test_norm_l.adapt(X_test)   # DON'T DO THIS! This is data leakage!
X_test_norm = test_norm_l(X_test)
```

**Why?** If you recompute μ and σ on test data, you're using information from the test set, which is cheating!

### Normalization vs. Vector Normalization

**Data Normalization (ML)** vs **Vector Normalization (Linear Algebra)** - these are DIFFERENT concepts:

| Aspect | Data Normalization (ML) | Vector Normalization (Linear Algebra) |
|--------|-------------------------|--------------------------------------|
| **Purpose** | Scale features to similar ranges | Make vector length = 1 |
| **Formula** | `(x - μ) / σ` | `v / ||v||` |
| **Output** | Centered at 0, std ≈ 1 | Unit vector (length = 1) |
| **Used for** | Preprocessing data for ML | Geometry, directions, orthogonal bases |
| **In this lab** | Yes! Used before training | No |

---

## Machine Learning Context

### Problem Type
**Binary Classification**: Predict one of two classes (good or bad roast)

### Input-Output Mapping

| Variable | Type | Shape | Meaning |
|----------|------|-------|---------|
| **X** | Input/Features | (200, 2) | What you measure (temperature, duration) |
| **Y** | Output/Labels | (200, 1) | What you want to predict (good=1 or bad=0 roast) |

### What the Neural Network Learns

The neural network learns the **decision boundary** that separates good roasts from bad roasts in the 2D feature space (temperature × duration).

Given new roasting parameters:
```python
new_roast = np.array([[230, 13.5]])  # 230°C for 13.5 minutes
prediction = model.predict(new_roast)  # → probability of being a good roast
```

---

## Visualization

When you plot the data with `plt_roast(X, Y)`, you'll see:
- **Red X's**: Good roasts (Y=1) - clustered in the valid region
- **Blue circles**: Bad roasts (Y=0) - outside the valid region
- **Purple lines**: The boundary constraints that define good roasts

This visualization helps understand:
1. The **decision boundary** the model needs to learn
2. How **non-linear** the problem is (requires a neural network, not just linear regression)
3. The **distribution** of good vs bad roasts in the parameter space

---

## Key Insights

### Why Neural Networks?

The decision boundary for this problem is **non-linear** and **bounded**:
- It's not a simple straight line
- Good roasts exist in a specific region with multiple constraints
- A neural network can learn complex decision boundaries that linear models cannot

### Real-World Context

This synthetic dataset mimics real coffee roasting, where:
- **Too cold or too hot** → bad roast
- **Too short or too long** → bad roast
- There's a **"sweet spot"** of temperature-duration combinations

This is a common pattern in real-world problems where optimal conditions exist within constrained ranges.

---

## Code Reference

The dataset generation code can be found in:
```python
# File: lab_coffee_utils.py
def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    # ... generates 200 random samples with labels
    return (X, Y.reshape(-1,1))
```

---

_Last updated: November 3, 2025_

