import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta ,date
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import time
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly.subplots import make_subplots
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


# Function to fetch historical stock data
#@st.cache_data(ttl=3600)  # cache expires in 1 hour
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to fetch and display company information
#@st.cache_data(show_spinner=False)
def display_company_info(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
    st.sidebar.write(f"**Name:** {info.get('longName', 'N/A')}")

    with st.expander("📊 Company Information"):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            st.subheader("📊 Company Information")
            st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
            st.markdown(f"**Country:** {info.get('country', 'N/A')}")
            st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
            st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
            st.markdown(f"**Summary:** {info.get('longBusinessSummary', 'N/A')}")
            
        except Exception as e:
            st.error(f"Error fetching company info: {e}")

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

def prophet_forecast(df):
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return m, forecast

# Streamlit app
def main():
    st.title('Stock Price Prediction')
    st.sidebar.header('Stock Price Prediction')

    # Company selection
    symbol = st.selectbox('Select Company', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'FB', 'AMZN', 'BTC-USD', 'ETH-USD', 'IBM', 'NESN.SW', 'SONY', 'SMSN.IL', 'HP', 'KO', 'TATAMOTORS.NS', 'WIPRO.NS', 'INFY', 'DELL', 'NOK', 'NVDA', 'BRK-B', 'JPM', 'V', 'MA', 'UNH', 'HD', 'PG', 'DIS', 'PFE', 'MRK', 'XOM', 'CVX', 'BAC', 'WMT', 'COST', 'BA', 'CAT', 'ORCL', 'CRM', 'T', 'VZ', 'GS', 'GE', 'MCD', 'SBUX', 'NKE', 'MDT', 'TXN', 'LLY', 'BMY', 'ABT', 'AMAT', 'QCOM', 'ADP', 'INTU', 'ZM', 'PYPL', 'F', 'GM', 'RIVN', 'LCID', 'SNAP', 'UBER', 'LYFT', 'PANW', 'SNOW', 'PLTR', 'DOCU', 'NET', 'SQ', 'SHOP', 'TWLO', 'CRWD', 'ROKU', 'DASH', 'ABNB', 'COIN', 'TSM', 'ASML', 'BABA', 'JD', 'NIO', 'BIDU', 'PDD', 'MELI', 'SE', 'HDB', 'RELIANCE.NS', 'ITC.NS', 'HCLTECH.NS', 'BAJAJFINSV.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'BHARTIARTL.NS', 'ADANIENT.NS', 'HINDUNILVR.NS', 'MARUTI.NS', 'ULVR.L', 'BP.L', 'SHEL', 'RIO', 'BHP', 'LMT', 'NOC', 'RTX', 'HON', 'DE', 'CSCO', 'MU', 'WBA', 'KDP', 'STZ', 'MO', 'ADM', 'GIS', 'KR', 'AFL', 'ADBE', 'SAP', 'TEAM', 'WDAY', 'ZS', 'TTD', 'MDB', 'DDOG', 'OKTA', 'FSLY', 'ESTC', 'CHWY', 'BB', 'BID', 'WIX', 'ETSY', 'NTDOY', 'MTCH', 'SPOT', 'RBLX', 'U', 'ADSK', 'NOW', 'HUBS', 'ZI', 'PARA', 'VIAC', 'LEN', 'DHI', 'PHM', 'NVR', 'TOL', 'EXPE', 'BKNG', 'HLT', 'MAR', 'LUV', 'DAL', 'UAL', 'AAL', 'CZR', 'MGM','LT.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ASIANPAINT.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'TITAN.NS','RELIANCE.NS', 'ITC.NS', 'INFY.NS', 'TCS.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ICICIBANK.NS', 'SBIN.NS','AXISBANK.NS', 'KOTAKBANK.NS', 'POWERGRID.NS', 'ONGC.NS', 'NTPC.NS', 'COALINDIA.NS', 'ADANIENT.NS', 'ADANIGREEN.NS','ADANIPORTS.NS', 'ADANIPOWER.NS', 'ADANITRANS.NS', 'HINDUNILVR.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'MARUTI.NS','MM.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TATAPOWER.NS', 'TATACHEM.NS', 'TATAELXSI.NS', 'JSWSTEEL.NS', 'BPCL.NS','IOC.NS', 'HINDALCO.NS', 'VEDL.NS', 'BRITANNIA.NS', 'DABUR.NS', 'NESTLEIND.NS', 'GAIL.NS', 'DRREDDY.NS', 'SUNPHARMA.NS','CIPLA.NS', 'LUPIN.NS', 'DIVISLAB.NS', 'BIOCON.NS','500325.BO', '500570.BO', '500180.BO', '532174.BO', '500209.BO', '500875.BO','500182.BO', '500696.BO', '500312.BO', '532540.BO', '500124.BO', '532215.BO','532155.BO', '500403.BO', '500087.BO', '500480.BO', '532281.BO', '532648.BO','500112.BO', '500010.BO', '500547.BO', '532977.BO'])

    # Year selection using a slider
    current_year = datetime.now().year
    year = st.slider('Select Year', min_value=2010, max_value=current_year, value=current_year)

    # Fetching stock data for the selected year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    #display company info
    display_company_info(symbol)

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
    # Calculate days required to get profit after prediction
    if profit > 0:
        st.sidebar.write("Forecasting 1 day ahead")
        days_to_profit = (stock_data.index[-1] - stock_data.index[-2]).days
        st.sidebar.write(f"Days required to get profit after prediction: {days_to_profit}")
    st.sidebar.write(f"Recommendation: {recommendation}")
    st.sidebar.write(f"Price before action: {price_before}")
    st.sidebar.write(f"Price after action: {price_after}")
    st.sidebar.write(f"Profit: {profit}")

    # Plotting actual and predicted closing prices
    with st.expander("📈 Stock Data Overview"):
        st.write("Stock Data Head Of Selected Year:", stock_data.head(5))
        st.write("Stock Data Tail Of Selected Year:", stock_data.tail(5))

    # Display Price 
    st.write("Predicted Price:", predicted_price1[0])

    # select days for display 
    num_days = st.slider('Number of Days to Display', min_value=1, max_value=len(stock_data), value=30)
    
   # Extract the correct closing price series
    try:
        if isinstance(stock_data['Close'], pd.DataFrame):
            close_series = stock_data['Close'][symbol]
        else:
            close_series = stock_data['Close']
    except Exception as e:
        st.error(f"Error extracting closing prices: {e}")
        st.stop()

    # Ensure index is aligned to the Series
    try:
        x_vals = close_series.tail(num_days).index.tolist()
        y_vals = close_series.tail(num_days).tolist()
        # Get the last actual closing price
        actual_last_price = close_series.iloc[-1]
    except Exception as e:
        st.error(f"Error preparing plot data: {e}")
        st.stop()

    # Plot
    trace_actual = go.Scatter(
        x=close_series.tail(num_days).index,
        y=close_series.tail(num_days),
        #x=stock_data.index[-num_days:],  # Already a DatetimeIndex
        #y=close_series.tail(num_days).tolist(),
        mode='lines+markers',
        name='Actual Closing Price',
        line=dict(color='blue'),
        hovertemplate='%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    )

    trace_predicted = go.Scatter(
        x=[stock_data.index[-1], stock_data.index[-1] + timedelta(days=1)],
        y=[actual_last_price, predicted_price1[0]],
        mode='markers+lines',
        name='Predicted Closing Price',
        marker=dict(color='red')
    )

    # Add current time to the graph
    current_time = datetime.now().strftime('%H:%M:%S')
    layout = go.Layout(
        title='📉 Stock Price Prediction',
        xaxis=dict(title='Date', rangeslider=dict(visible=True), type='date'),  # ⬅️ Add range slider
        yaxis=dict(title='Closing Price'),
        legend=dict(x=0, y=1),
        hovermode='x unified',  # ⬅️ Hover across all series for the same x
        annotations=[
                    dict(
                        x=stock_data.index[-1],
                        y=float(stock_data["Close"].iloc[-1]),
                        xref='x',
                        yref='y',
                        text=f'Last Actual Price: ${float(stock_data["Close"].iloc[-1]):.2f}',
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=-20,
                        xanchor='left',
                        yanchor='bottom',
                        bgcolor='white',
                        font=dict(color='black')
                    ),
                    dict(
                        x=stock_data.index[-1] + timedelta(days=1),
                        y=float(predicted_price1[0]),
                        xref='x',
                        yref='y',
                        text=f'Predicted Price: ${float(predicted_price1[0]):.2f}',
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=-20,
                        xanchor='left',
                        yanchor='bottom',
                        bgcolor='lightyellow',
                        font=dict(color='red')
                    )
                ]

    )

    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("📈 Prophet Forecast (30 Days Ahead)")
    model_prophet, forecast = prophet_forecast(stock_data)
    fig_prophet = plot_plotly(model_prophet, forecast)
    st.plotly_chart(fig_prophet)


main()
