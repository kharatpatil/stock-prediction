import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta ,date
import plotly.graph_objs as go
import time
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly.subplots import make_subplots
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to preprocess data and split into features and target
def preprocess_data(stock_data):
    # Check if the stock data is empty
    if stock_data.empty:
        return pd.DataFrame(), pd.Series()  # Return empty DataFrame and Series
    
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Close']
    return X, y

# Function to train the model
def train_model(X_train, y_train):
    if X_train.empty or y_train.empty:
        return None

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())  # Convert to 1D array
    return model

# Function to predict next day's closing price
def predict_price(model, last_data):
    if model is None or last_data.empty:
        return [0.0]  # Return 0.0 if model is None or last_data is empty

    predicted_price = model.predict(last_data)
    return predicted_price

# Function to generate recommendations based on predicted price
		
def generate_recommendations(predicted_price, current_price):
    predicted_price = float(predicted_price)
    current_price = float(current_price.iloc[0] if isinstance(current_price, pd.Series) else current_price)

    if predicted_price > current_price:
        recommendation = "Buy"
        price_before = current_price
        price_after = predicted_price
        profit = predicted_price - current_price
    else:
        recommendation = "Hold/Sell"
        price_before = current_price
        price_after = predicted_price
        profit = predicted_price - current_price

    return recommendation, price_before, price_after, profit


# Streamlit app
def main():
    st.title('Stock Price Prediction')
    st.sidebar.header('Stock Price Prediction')

    # Company selection
    symbol = st.selectbox('Select Company', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'FB', 'AMZN', 'BTC-USD', 'ETH-USD','IBM','NESN.SW','SONY','SMSN.IL','HP','KO','TATAMOTORS.NS','WIPRO.NS','INFY','DELL','NOK'])

    # Year selection using a slider
    current_year = datetime.now().year
    year = st.slider('Select Year', min_value=2010, max_value=current_year, value=current_year)

    # Fetching stock data for the selected year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Check if stock data is empty
    if stock_data.empty:
        st.error("No data found for the selected stock or date range. Please try again.")
        return

    # Preprocessing data
    X, y = preprocess_data(stock_data)

    # Ensure there is enough data
    if len(stock_data) < 2:
        st.error("Not enough data for the selected stock or date range. Please choose a different stock or date range.")
        return

    # Handle missing data (drop rows with NaN values)
    stock_data = stock_data.dropna()

    # Splitting data into training and testing sets
    test_size = 0.2 if len(stock_data) > 5 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Training the model
    model = train_model(X_train, y_train)
    if model is None:
        st.error("Model training failed. Not enough data to train.")
        return

    # Evaluating the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.sidebar.write(f'Training Score: {train_score}')
    st.sidebar.write(f'Testing Score: {test_score}')

    # Predicting next day's closing price
    last_data = X.tail(1)
    predicted_price1 = predict_price(model, last_data)

    st.sidebar.write(f"Predicted closing price for the next day: {predicted_price1[0]}")

    # Generate recommendations
    current_price = stock_data['Close'].iloc[-1]
    recommendation, price_before, price_after, profit = generate_recommendations(predicted_price1[0], current_price)
    st.sidebar.write(f"Recommendation: {recommendation}")
    st.sidebar.write(f"Price before action: {price_before}")
    st.sidebar.write(f"Price after action: {price_after}")
    st.sidebar.write(f"Profit: {profit}")

    # Calculate days required to get profit after prediction
    if profit > 0:
        days_to_profit = (stock_data.index[-1] - stock_data.index[-2]).days
        st.sidebar.write(f"Days required to get profit after prediction: {days_to_profit}")

    # Plotting actual and predicted closing prices
    st.write("Stock Data Tail:", stock_data['Close'].tail(5))  
    st.write("Predicted Price:", predicted_price1[0]) 
    num_days = st.slider('Number of Days to Display', min_value=1, max_value=len(stock_data), value=30)
    trace_actual = go.Scatter(x=stock_data.index[-num_days:], y=stock_data['Close'].tail(num_days), mode='lines', name='Actual Closing Price', line=dict(color='blue'))
    trace_predicted = go.Scatter(x=[stock_data.index[-1], stock_data.index[-1] + timedelta(days=1)], y=[stock_data['Close'].iloc[-1], predicted_price1[0]], mode='markers+lines', name='Predicted Closing Price', marker=dict(color='red'))
    # Add current time to the graph
    current_time = datetime.now().strftime('%H:%M:%S')
    layout = go.Layout(title='Stock Price Prediction', xaxis=dict(title='Date'), yaxis=dict(title='Closing Price'), legend=dict(x=0, y=1), annotations=[dict(x=stock_data.index[-1], y=stock_data['Close'].iloc[-1], xref='x', yref='y', text=f'Current Time: {current_time}', showarrow=True, arrowhead=7, ax=0, ay=-40)])

    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    st.plotly_chart(fig)

main()


