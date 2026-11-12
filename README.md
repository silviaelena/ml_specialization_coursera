# Machine Learning Specialization - Coursera

A personal learning repository containing Jupyter notebooks and exercises from Stanford's Machine Learning Specialization on Coursera.

## 📚 About

This repository serves as my learning playground for the Machine Learning Specialization course. It includes:
- Lab notebooks with hands-on exercises
- Personal notes and code comments
- Utility functions and helper scripts
- Experiments and explorations of ML concepts

> **Note**: This is a work in progress as I progress through the course.

## 📂 Repository Structure

```
.
├── Course1_SupervisedMachineLearning/
│   ├── Week1/
│   │   ├── Lessons_Python.md                         # Python lessons for the week
│   │   ├── C1_W1_Lab01_Python_Jupyter_Soln.ipynb     # Python & Jupyter intro
│   │   ├── C1_W1_Lab02_Model_Representation_Soln.ipynb # Linear regression model
│   │   ├── C1_W1_Lab03_Cost_function_Soln.ipynb      # Cost function
│   │   ├── C1_W1_Lab04_Gradient_Descent_Soln.ipynb   # Gradient descent
│   │   ├── lab_utils_common.py, lab_utils_uni.py     # Lab utilities
│   │   ├── data.txt, deeplearning.mplstyle
│   │   ├── images/                                   # Course images and diagrams
│   │   └── betaversion/                              # Older version of the Week 1 labs
│   └── Week2/
│       └── Lessons_Python.md                         # Python lessons for the week
└── Course2_AdvancedLearningAlgorithms/
    ├── C2_W1_Lab01_Neurons_and_Layers.ipynb        # Neural networks basics
    ├── C2_W1_Lab01_Neurons_and_Layers_Notes.md     # Lab 1 notes & explanations
    ├── C2_W1_Lab02_CoffeeRoasting_TF.ipynb         # TensorFlow implementation
    ├── C2_W1_Lab02_CoffeeRoasting_Notes.md         # Lab 2 notes (dataset, normalization)
    ├── C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb      # NumPy implementation
    ├── lab_coffee_utils.py                          # Coffee roasting utilities
    ├── lab_neurons_utils.py                         # Neural network utilities
    ├── lab_utils_common.py                          # Common utility functions
    ├── deeplearning.mplstyle                        # Matplotlib style file
    ├── images/                                      # Course images and diagrams
    │   ├── C2_W1_CoffeeRoasting.png
    │   ├── C2_W1_NeuronsAndLayers.png
    │   └── ...
    └── archive/                                     # Archived/older versions
        └── 20230202_lab_coffee_utils.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Required packages: NumPy, Matplotlib, TensorFlow

### Installation

1. Clone the repository:
```bash
git clone https://gitlab.cs.pub.ro/silvia_elena.nistor/ml_specialization_coursera.git
cd ml_specialization_coursera
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy matplotlib tensorflow jupyter
```

4. Launch Jupyter:
```bash
jupyter notebook
```

## 📝 Learning Notes

I maintain detailed notes on concepts I find particularly interesting or challenging in markdown files alongside each lab notebook. These notes include:
- Neural network fundamentals
- TensorFlow/Keras API patterns
- Dataset explanations (features, labels, decision boundaries)
- Data normalization and preprocessing
- Common pitfalls and their solutions

**Course 2 - Week 1 Notes:**
- [Lab 1: Neurons and Layers](Course2_AdvancedLearningAlgorithms/C2_W1_Lab01_Neurons_and_Layers_Notes.md)
- [Lab 2: Coffee Roasting with TensorFlow](Course2_AdvancedLearningAlgorithms/C2_W1_Lab02_CoffeeRoasting_Notes.md)

## 🎓 Course Information

- **Course**: [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- **Instructor**: Andrew Ng (Stanford University)
- **Platform**: Coursera

## 🔧 Development

Each notebook contains inline comments explaining concepts and implementations. Feel free to explore and modify the code for your own learning.

## 📜 License

This repository contains course materials from Coursera's Machine Learning Specialization. All original course content is property of Stanford University and Coursera. Personal notes and modifications are for educational purposes only.

## 🤝 Acknowledgments

- Stanford University and Coursera for providing the course materials
- Andrew Ng for creating this excellent learning resource
