import streamlit as st 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load, dump
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

st.title('📈 Machine Learning and Statistical Models Prediction')

model_options = ['None', 'Decision Tree Regressor', 'Linear Regressor', 'Random Forest Regressor', 'Support Vector Regressor']
model_choice = st.selectbox('Choose the Model:', model_options)

features = ['Open', 'High', 'Low', 'Volume', 'H-L', 'O-C', '7 DAYS MA', '14 DAYS MA', '21 DAYS MA', '7 DAYS STD DEV']

def preprocess_data(df):
    df = df.dropna().reset_index(drop=True)
    X = df[features]
    y = df['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)), df['Date']

# def preprocess_data_arima(df):
#     df = df.dropna().reset_index(drop=True)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)

#     feature_data = df[features]
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(feature_data)

#     pca = PCA(n_components=0.95)
#     pca_features = pca.fit_transform(scaled_features)

#     return pca_features, df['Close'], df.index, pca, scaler

def visualize_model_performance(dates, y_train, y_test, predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:len(y_train)], y=y_train, mode='markers', name='Train Actual', marker=dict(color='gray')))
    fig.add_trace(go.Scatter(x=dates[len(y_train):], y=y_test, mode='markers', name='Validation Actual', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines', name='Predictions', line=dict(color='orange', dash='dot')))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis_rangeslider_visible=True,
        width=1000  # Set the width to a larger value
    )
    return fig


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    all_predictions = np.concatenate([train_predictions, test_predictions])

    return train_rmse, test_rmse, train_mae, test_mae, all_predictions

def load_and_predict(model_choice):
    results = []
    bank_order = ['BBCA', 'BBNI', 'BBRI', 'BMRI']
    
    models = {}
    if model_choice == 'ARIMA':
        for bank in bank_order:
            model = load(f'{bank}_arima.joblib')
            df = pd.read_csv(f'{bank}_clean_modified.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            predictions = model.predict(n_periods=len(df))

            results.append({
                'Bank': bank,
                'Predictions': predictions
            })

            fig = visualize_model_performance(df.index, None, None, predictions, f'ARIMA Predictions for {bank}')
            st.plotly_chart(fig, use_container_width=True)
    else:
        models = {
            'Linear Regressor': {bank: load(f'lr_model_pca_{bank}.joblib') for bank in bank_order},
            'Random Forest Regressor': {bank: load(f'rf_model_pca_{bank}.joblib') for bank in bank_order},
            'Support Vector Regressor': {bank: load(f'{bank}_best_svr_model.joblib') for bank in bank_order},
            'Decision Tree Regressor': {bank: load(f'{bank}_best_dtr_model.joblib') for bank in bank_order}
        }.get(model_choice, {})

        for bank in models:
            model = models[bank]
            df = pd.read_csv(f'{bank}_clean_modified.csv')
            (X_train, X_test, y_train, y_test), dates = preprocess_data(df)

            pca = PCA(n_components=4)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            X_pca = np.concatenate([X_train_pca, X_test_pca])
            predictions = model.predict(X_pca)

            train_rmse, test_rmse, train_mae, test_mae, _ = evaluate_model(model, X_train_pca, X_test_pca, y_train, y_test)
            
            max_pred_index = np.argmax(predictions[len(y_train):])
            max_pred_value = predictions[len(y_train) + max_pred_index]
            max_pred_date = dates.iloc[len(y_train) + max_pred_index]
            actual_value_at_max_pred = y_test.iloc[max_pred_index]

            results.append({
                'Bank': bank,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Prediction': max_pred_value,
                'Date': max_pred_date,
                'Actual': actual_value_at_max_pred
            })
            
            fig = visualize_model_performance(dates, y_train, y_test, predictions, f'Prediction for {bank}')
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"### ⬆️ Highest Prediction Details for {bank}")
            prediction_details_df = pd.DataFrame([{
                "Prediction Value": max_pred_value,
                "Date": max_pred_date,
                "Actual Value": actual_value_at_max_pred
            }])
            st.table(prediction_details_df)

    return results


if model_choice in ['Linear Regressor', 'Random Forest Regressor', 'Support Vector Regressor', 'Decision Tree Regressor']:
    results = load_and_predict(model_choice)
    
    st.subheader('Model Performance Metrics')
    results_df = pd.DataFrame(results)

    # Change index to start from 1
    results_df.index += 1

    # Highlight the best model
    def highlight_best_row(row):
        if row.name == 1:
            return ['background-color: lightgreen; font-weight: bold'] * len(row)
        else:
            return [''] * len(row)

    # Style the dataframe
    styled_df = results_df.style.apply(highlight_best_row, axis=1).set_table_styles(
        [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
    )

    st.write(styled_df.to_html(), unsafe_allow_html=True)
