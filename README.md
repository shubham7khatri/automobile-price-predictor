# 🚘 Automobile Price Prediction Web App

An interactive machine learning web application that predicts automobile prices based on vehicle specifications using **Random Forest Regression** and **Linear Regression**. Built with Python and Streamlit, the application allows users to enter vehicle details, generate price predictions, and compare model performance through evaluation metrics.

## ✨ Features

- **Price Prediction:** Estimate automobile prices using trained regression models.
- **Multiple Models:** Choose between Random Forest Regression and Linear Regression.
- **Data Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features.
- **Machine Learning Pipelines:** Integrate preprocessing and prediction into reusable Scikit-learn pipelines.
- **Model Evaluation:** Compare models using R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- **Data Visualization:** View actual versus predicted prices to understand model performance.
- **Interactive Web Interface:** Enter vehicle specifications and get predictions directly in your browser.
- **Public Deployment:** Access the application online without installing Python or machine learning libraries.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Pandas & NumPy** — Data cleaning and manipulation
- **Scikit-learn** — Preprocessing, machine learning pipelines, regression models, and evaluation
- **Matplotlib** — Data visualization
- **Joblib** — Saving and loading trained models
- **Streamlit** — Interactive web application and deployment

## 🤖 Machine Learning Models

### 1. Random Forest Regression
An ensemble learning algorithm that combines multiple decision trees to predict automobile prices.

### 2. Linear Regression
A regression algorithm used as a baseline to estimate the relationship between vehicle features and prices.

Both models are evaluated on a held-out test dataset to compare their predictive performance.

## ⚙️ How It Works

1. Load and clean the automobile dataset.
2. Handle missing values and prepare numerical and categorical features.
3. Split the data into training and testing sets.
4. Build preprocessing and regression pipelines.
5. Train and evaluate both models.
6. Save the fitted pipelines using Joblib.
7. Deploy the Streamlit application for interactive predictions.

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/shubham7khatri/automobile-price-predictor
cd automobile-price-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models if the saved artifacts are not included:

```bash
python train_model.py
```

Launch the application:

```bash
streamlit run app.py
```

## 🌐 Live Demo

**Try the application:** (https://automobile-price-predictor-emtvhqfrjagvtg8vxdf2zw.streamlit.app/)

## 🎯 Project Objective

This project demonstrates practical machine learning skills, including data preprocessing, feature transformation, regression modeling, model evaluation, serialization, and deployment of a machine learning application.

It was developed as a hands-on project to explore how machine learning models can be transformed into interactive tools that users can access through a web browser.

## ⚠️ Disclaimer

Predicted prices are estimates based on patterns learned from the training dataset. Actual automobile market prices may differ.

---

**Developed using Python, Scikit-learn, and Streamlit.**
