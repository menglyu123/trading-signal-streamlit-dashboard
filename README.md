# Trading Signal Streamlit Dashboard

A real-time trading signal dashboard built with Streamlit that provides market predictions and analysis for HK and US stocks.

## Features

- Real-time market predictions for HK and US stocks
- Interactive stock analysis with technical indicators
- Historical prediction tracking
- Individual stock analysis with charts
- Market strength indicators

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Deployment to Streamlit Community Cloud

This dashboard is ready for deployment to Streamlit Community Cloud. The deployment will automatically:

1. Install all required dependencies from `requirements.txt`
2. Use the configuration in `.streamlit/config.toml`
3. Deploy the main dashboard application

### Deployment Steps:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository
5. Set the main file path to: `dashboard.py`
6. Click "Deploy"

## Project Structure

- `dashboard.py` - Main Streamlit application
- `auto_trade.py` - Trading logic and prediction algorithms
- `util.py` - Utility functions
- `data/` - Data processing and import modules
- `model/` - Machine learning models
- `backtest/` - Backtesting functionality
- `results/` - Generated results and data

## Dependencies

- Streamlit - Web framework
- Pandas - Data manipulation
- YFinance - Stock data
- TensorFlow - Machine learning
- Plotly/Matplotlib - Visualization
- And more (see requirements.txt)

## Notes

- The dashboard requires internet connection to fetch real-time stock data
- Some features may require additional API keys for enhanced functionality
- The application is optimized for cloud deployment with proper caching 