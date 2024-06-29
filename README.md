# Monelytics-Stock-Prediction

**About:** Monelitys is an AI prototype deployed on Streamlit as a web-based Python application designed for stock prediction purposes. Our team has trained it using a combination of machine learning and deep learning models. Monelytics can predict the closing price of the big 4 banks in Indonesia accurately. The prediction result are primarily measured by two main error metrics: RMSE (Root Mean Square Error) and MAE (Mean Absolute Error).

| Machine Learning           | Deep Learning |
| -------------------------- | ------------- |
| Linear Regression          | CNN           |
| Support Vector Regression  | ANN           |
| Random Forest Regression   | LSTM          |
| ARIMA                      | Prophet       |
| Decision Tree Regression   |               |

**Pre-processing before training the models:** This project requires at least 7 features. Since the original dataset only provides 6 features, we addressed this issue through feature engineering. We used data from 2019 to 2024 (Last updated in Feb) that splitting into 80% for training dan 20% for testing.

This project predicts the closing price based on the original prediction and compares which model provides the most accurate result overall. We found that the linear regression and support vector machine are the best models for predicting the close price of bank stock. On the other hand, CNNs are not suitable for stock prediction and have shown worse performance compared to the other models, based on observations of RMSE and MAE values. 

**Demonstration Video for Monelytics Web:** https://youtu.be/GRpjmhJcJbQ?si=HaiPP84lL96NXFO4

**Contributor:**
1. Dean Hans Felandio Saputra
2. Calvin Alexander
3. Marvella Shera Devi
4. Verren Angelina Saputra
