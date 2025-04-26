# 🎬 Movie Rating Prediction

## 📌 Project Overview

The **Movie Rating Prediction** project aims to build a machine learning model that accurately predicts movie ratings based on a variety of attributes. By leveraging advanced data preprocessing, feature engineering, robust model selection, and comprehensive performance evaluation, this project demonstrates a practical end-to-end machine learning pipeline.

---

## 🎯 Objective

- **Predict Movie Ratings:** Develop a model that accurately estimates movie ratings based on features such as:
  - **Votes:** Derived from user input (after removing formatting artifacts).
  - **Duration:** Converted from strings (e.g., "109 min") to numeric values.
  - **Director Success Rate:** A computed metric representing the average rating for movies by a given director.
  - **Genre Average Rating:** The average rating for movies in each genre.
- **Performance Evaluation:** Quantitatively assess the model using RMSE, R², and a custom accuracy metric (predictions within a ±0.5 margin of the actual rating), all visualized in an intuitive bar chart.

---

## 🛠️ Methodology & Approach

### 1. Data Preprocessing

- **Handling Missing Values:**  
  Missing values are managed by either imputing (using mean values) or dropping them to ensure clean, usable data.

- **Data Cleaning:**  
  - **Votes Conversion:** Remove commas from the `Votes` column and convert it to a numeric format.
  - **Duration Parsing:** Strip the " min" suffix from the `Duration` column and convert it into a float.

- **Feature Engineering:**  
  - Compute `Director_Success_Rate` by averaging ratings for each director.
  - Derive `Genre_Average_Rating` by grouping the data by genre.
  
- **Encoding & Scaling:**  
  - Use one-hot encoding for categorical fields if needed.
  - Apply `StandardScaler` to all numerical features.
  - **Alignment:** Ensure that both the training and test datasets have exactly the same columns (both in order and name) to avoid errors during prediction.

### 2. Model Training & Selection

- The project employs a **Random Forest Regressor** due to its efficiency and robustness in handling complex, non-linear relationships in data.
- The model is trained on the scaled training dataset and then used to predict ratings on the test set.

### 3. Performance Evaluation

- **Metrics Computed:**
  - **RMSE (Root Mean Squared Error):** Measures the average prediction error.
  - **R² Score:** Determines the proportion of variance explained by the model.
  - **Custom Accuracy:** Defined here as the percentage of predictions within a ±0.5 margin of the actual values.
  
- **Visualization:**  
  A bar chart is generated using `matplotlib` to visually compare the RMSE, R² Score, and Accuracy, making performance insights easily accessible.

---

## 🚀 How to Run This Project

### Prerequisites

- **Python 3.x**
- Git
- [pip](https://pip.pypa.io/en/stable/)

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/movie-rating-prediction.git
   cd movie-rating-prediction
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download/Place the Dataset:**
   - Place your dataset file (e.g., `movie_dataset.csv`) into the `data/` directory.

4. **Run the Application:**

   - From the project root, run:
     ```bash
     python src/movie_rating_prediction.py
     ```
   - Alternatively, open and run the Jupyter Notebook in the `notebooks/` directory for an interactive session.


---

## 📊 Results & Performance

When you run the project, the following outcomes are provided:

- **RMSE:** The root mean squared error of the predictions.
- **R² Score:** The percentage of variance in the ratings explained by the model.
- **Accuracy:** The custom metric showing the percentage of predictions within a ±0.5 margin of the true ratings.

A bar chart is rendered to visually summarize these metrics, giving an immediate sense of model performance.

---

## 🔍 Further Improvements

Future work may include:
- **Advanced Feature Engineering:** Incorporate more refined features (e.g., sentiment analysis of reviews).
- **Hyperparameter Tuning:** Optimize model parameters using GridSearchCV or RandomizedSearchCV.
- **Ensemble Methods:** Combine predictions from different models for improved robustness.

---
