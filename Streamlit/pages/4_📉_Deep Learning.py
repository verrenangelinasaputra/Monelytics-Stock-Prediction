import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from joblib import dump
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import yfinance as yf

# Preprocess data for Deep Learning models
def preprocess_data(df):
    df = df.dropna().reset_index(drop=True)
    features = df[['Open', 'High', 'Low', 'Volume', 'H-L', 'O-C', '7 DAYS MA', '14 DAYS MA', '21 DAYS MA', '7 DAYS STD DEV']]
    target = df['Close']
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return train_test_split(features, target, test_size=0.2, random_state=42), df['Date']

# Preprocess data for ARIMA model
def preprocess_data_arima(df):
    df = df.dropna().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Feature selection and scaling
    feature_data = df[['Open', 'High', 'Low', 'Volume', 'H-L', 'O-C', '7 DAYS MA', '14 DAYS MA', '21 DAYS MA', '7 DAYS STD DEV']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    pca_features = pca.fit_transform(scaled_features)

    return pca_features, df['Close'], df.index, pca, scaler

# Visualize model performance for ARIMA and Deep Learning models
def visualize_model_performance(dates, y_train, y_test, y_train_pred, y_test_pred, title):
    train_dates = dates[:len(y_train)]
    test_dates = dates[len(y_train):len(y_train) + len(y_test)]
    all_dates = dates[:len(y_train) + len(y_test)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_dates, y=y_train, mode='markers', name='Train Actual', marker=dict(color='gray')))
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='markers', name='Test Actual', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=all_dates, y=np.concatenate((y_train_pred, y_test_pred)).flatten(), mode='lines', name='Predictions', line=dict(color='orange', width=2, dash='dot')))

    fig.update_layout(
        title=title,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        yaxis=dict(title='Close Price IDR')
    )
    return fig

# Evaluate ARIMA model
def evaluate_arima_model(train, test, order):
    history = [x for x in train]
    predictions_train = []
    predictions_test = []

    # Training phase
    for t in range(len(train)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions_train.append(yhat)
        history.append(train[t])

    # Testing phase
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions_test.append(yhat)
        history.append(test[t])

    return model_fit, predictions_train, predictions_test

# Load and predict using pre-trained Deep Learning model
def load_and_predict_model(df, model_path, dataset_name, model_type):
    (X_train, X_test, y_train, y_test), dates = preprocess_data(df)

    if model_type == 'CNN':
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = load_model(model_path)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    fig = visualize_model_performance(dates, y_train, y_test, train_predictions, test_predictions, f'{model_type} Model for {dataset_name}')
    st.plotly_chart(fig, use_container_width=True)

    max_pred_index = np.argmax(test_predictions)
    max_pred_value = test_predictions[max_pred_index]
    max_pred_date = dates.iloc[len(y_train) + max_pred_index]
    actual_value_at_max_pred = y_test.iloc[max_pred_index]

    prediction_details_df = pd.DataFrame([{
        "Prediction Value": max_pred_value,
        "Date": max_pred_date,
        "Actual Value": actual_value_at_max_pred
    }])

    return {
        'Bank': dataset_name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Prediction': max_pred_value,
        'Date': max_pred_date,
        'Actual': actual_value_at_max_pred,
        # 'Details': prediction_details_df
    }

# Process ARIMA model
def process_arima(df, dataset_name):
    pca_features, close_prices, dates, pca, scaler = preprocess_data_arima(df)
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices[:train_size], close_prices[train_size:]
    order = (5, 1, 0)
    model_fit, predictions_train, predictions_test = evaluate_arima_model(train, test, order)

    train_rmse = np.sqrt(mean_squared_error(train, predictions_train))
    train_mae = mean_absolute_error(train, predictions_train)
    test_rmse = np.sqrt(mean_squared_error(test, predictions_test))
    test_mae = mean_absolute_error(test, predictions_test)

    all_predictions_train = np.concatenate([predictions_train, [np.nan]*len(test)])
    all_predictions_test = np.concatenate([[np.nan]*len(train), predictions_test])

    fig = visualize_model_performance(dates, train, test, all_predictions_train, all_predictions_test, f'ARIMA Model with PCA for {dataset_name}')
    st.plotly_chart(fig, use_container_width=True)

    # Save models and scalers
    dump(pca, f'{dataset_name}_pca.joblib')
    dump(scaler, f'{dataset_name}_scaler.joblib')
    dump(model_fit, f'{dataset_name}_arima.joblib')

    return {
        'Dataset': dataset_name,
        'ARIMA_Train_RMSE': train_rmse,
        'ARIMA_Train_MAE': train_mae,
        'ARIMA_Test_RMSE': test_rmse,
        'ARIMA_Test_MAE': test_mae
    }

# Preprocess and predict using Prophet model
def preprocess_and_predict_prophet(csv_file, future_period=365):
    df = pd.read_csv(csv_file)
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df['Close']

    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05, seasonality_mode='multiplicative')
    model.fit(df)

    future = model.make_future_dataframe(periods=future_period, include_history=True)
    future = future[future['ds'] <= df['ds'].max()]

    forecast = model.predict(future)
    forecast_filtered = forecast[['ds', 'yhat']].merge(df[['ds', 'y']], on='ds')

    mae = mean_absolute_error(forecast_filtered['y'], forecast_filtered['yhat'])
    rmse = mean_squared_error(forecast_filtered['y'], forecast_filtered['yhat'], squared=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Actual Data', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predictions', line=dict(color='red')))

    fig.update_layout(
        title='Stock Price Prediction with Prophet',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        yaxis=dict(autorange=True, fixedrange=False)
    )

    st.plotly_chart(fig, use_container_width=True)

    return {
        'Dataset': csv_file.split('_')[0],
        'Prophet_MAE': mae,
        'Prophet_RMSE': rmse
    }

# Streamlit UI Components
st.title("📉 Deep Learning and Statistical Models Prediction")

model_options = ['None', 'ANN', 'CNN', 'LSTM', 'Prophet']
model_choice = st.selectbox('Choose the Model:', model_options)

if model_choice in ['ANN', 'CNN', 'LSTM', 'Prophet']:
    banks = ['BBCA', 'BBNI', 'BBRI', 'BMRI']
    results = []

    for bank in banks:
        data = pd.read_csv(f'{bank}_clean_modified.csv')  # Ensure data path is correct
        if model_choice == 'ANN':
            model_path = f'{bank}_ANN (1).h5'  # Path to pre-trained ANN model
            result = load_and_predict_model(data, model_path, bank, model_choice)
        elif model_choice == 'CNN':
            model_path = f'{bank}_CNN.h5'  # Path to pre-trained CNN model
            result = load_and_predict_model(data, model_path, bank, model_choice)
        elif model_choice == 'LSTM':
            model_path = f'{bank}_CNN.h5'  # Path to pre-trained CNN model
            result = load_and_predict_model(data, model_path, bank, model_choice)
        elif model_choice == 'Prophet':
            result = preprocess_and_predict_prophet(f'{bank}_clean_modified.csv')
        results.append(result)

        # Display plot and prediction details for each bank
        st.write(f"### ⬆️ Highest Prediction Details for {bank}")
        st.table(result)

    # Display results
    st.subheader("Model Performance Metrics for All Banks")
    results_df = pd.DataFrame(results)

    # Sort results by 'Test RMSE' if available, otherwise by 'RMSE'
    if 'Test RMSE' in results_df.columns:
        results_df = results_df.sort_values(by='Test RMSE')
    else:
        results_df = results_df.sort_values(by='Prophet_RMSE')

    # Reset index to start from 1
    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1

    # Highlight the best model
    def highlight_best_row(row):
        if row.name == results_df.index[0]:
            return ['background-color: lightgreen; font-weight: bold'] * len(row)
        else:
            return [''] * len(row)

    # Style the dataframe
    styled_df = results_df.style.apply(highlight_best_row, axis=1).set_table_styles(
        [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
    )

    st.write(styled_df.to_html(), unsafe_allow_html=True)
