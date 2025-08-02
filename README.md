# HyperParameter-Tuning
{Completed Hyperparameter Tuning) I fine-tuned the model's performance by experimenting with different hyperparameters such as learning rate, number of layers, activation functions, optimizers, and dropout rates. The tuning significantly improved accuracy and reduced overfitting.


# 🩺 Diabetes Prediction Model with Hyperparameter Tuning

This project builds a machine learning model to predict the likelihood of diabetes in individuals using the **Pima Indians Diabetes Dataset**. It includes hyperparameter tuning to optimize model performance and minimize overfitting.


## 📁 Files

- `HyperParameterTuning.ipynb` – Jupyter/Colab notebook with model training and tuning logic.
- `diabetes.csv` – Input dataset with patient features and outcomes.
- `README.md` – Project documentation.



## 📊 Dataset Overview

- **Source**: Pima Indians Diabetes Dataset (UCI ML Repository / Kaggle)
- **Records**: 768 patients
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = No diabetes, 1 = Diabetes)


## 🧠 Model Overview

- Model: ANN
- Preprocessing steps:
  - Handling missing or zero values
  - Feature scaling (e.g., StandardScaler)
- Evaluation metrics: Accuracy, Precision, Recall, F1-score



## 🔧 Hyperparameter Tuning

Performed tuning to optimize:
- Learning rate
- Number of layers/units (if using NN)
- Activation functions
- Optimizer (SGD, Adam, etc.)
- Dropout rate
- Regularization (if used)

### 📈 Best Results:
- Accuracy: 77.59%
- Tuned Parameters:
 'num_layers': 1,
 'units0': 112,
 'activation0': 'relu',
 'dropout0': 0.8,
 'optimizer': 'adam'

## 📌 How to Run

1. Open [`HyperParameter-Tuning.ipynb`](https://colab.research.google.com/drive/1BXzguLJKLQznr3Tiw-kUMdNCxxyCOSJE?usp=sharing) in Colab or Jupyter.
2. Make sure `diabetes.csv` is present in the same directory or uploaded in Colab.
3. Run all cells step-by-step.
4. Modify hyperparameters to experiment with performance.



## 🔍 Future Improvements

- Use automated tuning (e.g., Keras Tuner, Optuna)
- Try different models like XGBoost or LightGBM
- Build a Streamlit or Flask app for real-time prediction


## 💬 Contact

Created by Raja Rathour
Feel free to reach out for collaboration or questions!



