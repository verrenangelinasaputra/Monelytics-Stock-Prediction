import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

# Set current and past date range
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

bank_list = ['BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK']
company_names = ["BCA", "BRI", "BNI", "Mandiri"]

# Prepare to store data and compute maximums
bank_data = {}
max_closes = []
max_volumes = []
max_opens = []

# Download data for each bank and calculate maximum values
for bank, name in zip(bank_list, company_names):
    print(f"Downloading data for {bank}")  # Debugging print
    try:
        data = yf.download(bank, start, end)
        if data.empty:
            st.error(f"No data returned for {bank}. Please check the ticker.")
            continue
        data.reset_index(inplace=True)
        data['company_name'] = name
        max_close = data['Close'].max()
        max_volume = data['Volume'].max()
        max_open = data['Open'].max()
        max_closes.append({'Bank': name, 'Max Close Price': max_close})
        max_volumes.append({'Bank': name, 'Max Volume': max_volume})
        max_opens.append({'Bank': name, 'Max Open Price': max_open})
        bank_data[name] = data
    except Exception as e:
        st.error(f"Failed to download data for {bank}: {e}")
        continue

# Combine into one DataFrame
df = pd.concat(bank_data.values(), axis=0) if bank_data else pd.DataFrame()

# DataFrame untuk harga close dan volume maksimal
visualization_close = pd.DataFrame(max_closes)
visualization_volume = pd.DataFrame(max_volumes)
visualization_close = visualization_close.sort_values(by='Max Close Price', ascending=False)
visualization_volume = visualization_volume.sort_values(by='Max Volume', ascending=False)

# Streamlit UI Components
st.title("🔍 Dashboard")

# Plot pertama: Harga close harian
option = st.radio(
    'Choose the Bank for Daily Close Price:',
    company_names,
    format_func=lambda x: x,
    index=0,
    horizontal=True
)

selected_bank_data = df[df['company_name'] == option]
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=selected_bank_data['Date'], y=selected_bank_data['Close'], mode='lines', name='Close'))
fig1.update_layout(
    title=f'Stock Price Visualization {option}',
    xaxis=dict(rangeslider=dict(visible=True), type='date')
)
st.plotly_chart(fig1, use_container_width=True)

# Plot kedua: Harga close maksimum
fig2 = go.Figure(data=[
    go.Bar(name='Max Close Price', x=visualization_close['Bank'], y=visualization_close['Max Close Price'])
])
fig2.update_layout(
    title='Max Close Price of Each Bank in the Last Year',
    xaxis_title="Bank",
    yaxis_title="Max Close Price",
    barmode='group'
)
st.plotly_chart(fig2, use_container_width=True)

# Deskripsi dengan latar belakang hijau muda
description = """
Over the past year, BCA (Bank Central Asia) has shown the best performance with the highest closing price compared to other banks in Indonesia, indicating a positive market perception and strong potential fundamentals. On the other hand, Bank Mandiri, BRI, and BNI offer lower prices, which may appeal to investors looking for value or growth not fully recognized by the market.
"""
st.markdown(f"<div style='background-color:lightgreen; padding: 10px;'>{description}</div>", unsafe_allow_html=True)

st.write("###")
st.write("###")

# Plot ketiga: Visualisasi volume penjualan saham
volume_option = st.radio(
    'Choose the Bank for Sales Volume Visualization:',
    company_names,
    format_func=lambda x: x,
    index=0,
    horizontal=True
)

selected_volume_data = df[df['company_name'] == volume_option]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=selected_volume_data['Date'], y=selected_volume_data['Volume'], mode='lines', name='Volume'))
fig3.update_layout(
    title=f'Sales Volume Visualization {volume_option}',
    xaxis=dict(rangeslider=dict(visible=True), type='date')
)
st.plotly_chart(fig3, use_container_width=True)

# Plot keempat: Volume penjualan maksimum
fig4 = go.Figure(data=[
    go.Bar(name='Max Volume', x=visualization_volume['Bank'], y=visualization_volume['Max Volume'])
])
fig4.update_layout(
    title='Maximum Sales Volume of Each Bank in the Last Year',
    xaxis_title="Bank",
    yaxis_title="Max Volume",
    barmode='group'
)
st.plotly_chart(fig4, use_container_width=True)

description = """
BRI exhibits the largest stock sales volume but has a relatively low close price, which may suggest several underlying issues. High trading volume coupled with a low price often indicates a lack of confidence among investors or responses to negative news impacting the company, leading to sell-offs. This scenario also reflects high liquidity, allowing shares to be traded easily without significant price changes. However, if prices remain low despite high volumes, it could indicate a lack of demand at higher price levels or a general market undervaluation of the stock. Furthermore, significant sales by major shareholders can trigger high volumes and price drops, particularly if the market cannot absorb the sales at stable prices.
"""
st.markdown(f"<div style='background-color:lightgreen; padding: 10px;'>{description}</div>", unsafe_allow_html=True)

st.write("###")
st.write("###")

# Plot kelima: Visualisasi harga buka
open_option = st.radio(
    'Choose the Bank for Open Price Visualization:',
    company_names,
    format_func=lambda x: x,
    index=0,
    horizontal=True
)

selected_open_data = df[df['company_name'] == open_option]
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=selected_open_data['Date'], y=selected_open_data['Open'], mode='lines', name='Open'))
fig5.update_layout(
    title=f'Open Price Visualization {open_option}',
    xaxis=dict(rangeslider=dict(visible=True), type='date')
)
st.plotly_chart(fig5, use_container_width=True)


visualization_open = pd.DataFrame(max_opens).sort_values(by='Max Open Price', ascending=False)
fig6 = go.Figure(data=[go.Bar(name='Max Open Price', x=visualization_open['Bank'], y=visualization_open['Max Open Price'])])
fig6.update_layout(title='Maximum Open Price of Each Bank in the Last Year', xaxis_title="Bank", yaxis_title="Max Open Price", barmode='group')
st.plotly_chart(fig6, use_container_width=True)



banks = dict(zip(company_names, bank_list))

# Prepare to store data
bank_data = {}

# Function to download and calculate moving averages
def download_and_calculate_moving_averages(ticker):
    data = yf.download(ticker, start, end)
    ma_days = [10, 20, 50]
    for ma in ma_days:
        data[f'MA for {ma} days'] = data['Adj Close'].rolling(window=ma).mean()
    return data

option = st.radio('Choose the Bank:', company_names, format_func=lambda x: x, index=0, horizontal=True)

# Check if data has already been downloaded
if option not in bank_data:
    bank_data[option] = download_and_calculate_moving_averages(banks[option])

# Access the selected bank data
selected_data = bank_data[option]

# Plotting using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['Adj Close'], mode='lines', name='Adj Close'))
fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['MA for 10 days'], mode='lines', name='7-Day MA'))
fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['MA for 20 days'], mode='lines', name='14-Day MA'))
fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['MA for 50 days'], mode='lines', name='21-Day MA'))

fig.update_layout(
    title=f'{option} Stock Prices with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Adjusted Close Price',
    xaxis=dict(
        rangeslider=dict(visible=True),
        type='date'
    )
)

st.plotly_chart(fig, use_container_width=True)

description = """
BCA and Mandiri exhibit strong trends but experience significant volatility, which may indicate reactions to external factors or market events. Then, BRI and BNI show strong bullish trends but also experience declines, suggesting that investors may need to be cautious of changing trends. All banks demonstrate significant levels of volatility at various points, which is important for investors to consider when planning entry or exit strategies.
"""
st.markdown(f"<div style='background-color:lightgreen; padding: 10px;'>{description}</div>", unsafe_allow_html=True)


