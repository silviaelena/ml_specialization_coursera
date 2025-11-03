# ML_Specialization_Coursera

## Disclaimer: Work in Progress

## Getting started

This is a playground for Jupyter Notebooks used in Coursera Machine Learning Specialization from Stanford.
It is meant as a beginner work for just ramping up in the world of ML. I will also add occasional notes for different notions/areas/subjects.

For each cell of code where there is something I need some notes on, I add comments.

## Notes on Neural Networks and TensorFlow/Keras

### Weight Initialization vs. Training

A common point of confusion for me when starting with TensorFlow/Keras was understanding what happens automatically vs. what requires training.

#### 1. Initialization (Automatic)
When you first pass data through a layer, Keras **randomly initializes** the weights and sets bias to zero:

```python
linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
a1 = linear_layer(X_train[0].reshape(1,1))  # Triggers initialization

w, b = linear_layer.get_weights()
print(w, b)  # w = [[-1.09]], b = [0.]  ← Random initial values!
```

These are **random starting values**, not the optimal ones!

#### 2. Training (You Must Do This)
To find the **optimal** weights and biases that fit your data, you need to **train** the model:

```python
# Compile the model (specify loss function and optimizer)
model.compile(loss='mean_squared_error', optimizer='sgd')

# Train the model (this learns the optimal weights!)
model.fit(X_train, Y_train, epochs=100)

# Now the weights are optimized to fit your data
```

#### What Each Part Does

| Step | What Happens | Who Does It |
|------|--------------|-------------|
| **Create layer** | Define architecture | You |
| **Initialize weights** | Set random starting values | Keras (automatic) |
| **Train model** | Learn optimal weights from data | You call `model.fit()` |
| **Make predictions** | Use learned weights | You call `model.predict()` |

#### Real Workflow Example

```python
# 1. Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='linear')
])

# 2. Compile (specify how to learn)
model.compile(optimizer='adam', loss='mse')

# 3. Train (THIS is where optimal weights are computed!)
model.fit(X_train, Y_train, epochs=1000)

# 4. The model now has learned the best weights for your data
```

**Key Point**: Keras initializes random weights automatically, but **you must train the model** for it to learn the optimal weights that fit your data. Training is the process where the model adjusts weights to minimize error on your training data.

### Defining Neural Networks Model Input: Old vs. New API

You might encounter a deprecation warning about `input_shape` or `input_dim`. The TensorFlow/Keras API has evolved toward a more explicit way of defining model inputs.

#### Old Way (Deprecated, But Still Works)

```python
model = Sequential([
    Dense(1, input_dim=1, activation='sigmoid')  # Triggers warning
])
```

#### New Recommended Way

```python
from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(1,)),                           # Explicit Input layer
    Dense(1, activation='sigmoid')
])
```

#### Why the Change?

**1. Explicit is Better Than Implicit**
Having a dedicated `Input` layer makes it crystal clear what shape your model expects:

```python
# Old: input shape hidden in Dense layer parameters
Dense(128, input_shape=(784,))

# New: input shape is explicit and separate
Input(shape=(784,))
Dense(128)
```

**2. Consistency Across APIs**
TensorFlow has multiple ways to build models (Sequential, Functional, Subclassing). Using `Input` works consistently across all of them:

```python
# Functional API - always used Input
inputs = Input(shape=(784,))
x = Dense(128)(inputs)
model = Model(inputs, x)

# Sequential API - now uses Input too for consistency
model = Sequential([
    Input(shape=(784,)),
    Dense(128)
])
```

**3. Better Model Inspection**
With an explicit `Input` layer, you can inspect the model architecture more clearly with `model.summary()`.

#### Does the Old Way Still Work?

**Yes!** It's just deprecated, not removed. The course notebooks use the older approach since they were created before this change. Your code will work fine, you'll just see warnings.

**Bottom line**: It's a stylistic/API evolution for better code clarity. Both ways functionally do the same thing, but the new way is more explicit and consistent.
