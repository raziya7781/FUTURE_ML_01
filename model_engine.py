import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class SalesForecaster:
    def __init__(self, data_path='historical_sales.csv'):
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
    def get_category_data(self, category=None):
        if category and category != 'All':
            return self.df[self.df['Category'] == category]
        return self.df

    def prepare_data(self, df):
        # Prophet expects columns 'ds' and 'y'
        prophet_df = df.groupby('Date')['Total_Sales'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        return prophet_df

    def get_advanced_features(self, df):
        """Extracts time-based features for analysis."""
        feat_df = df.copy()
        feat_df['Month'] = feat_df['Date'].dt.month
        feat_df['Year'] = feat_df['Date'].dt.year
        feat_df['DayOfWeek'] = feat_df['Date'].dt.dayofweek
        feat_df['IsWeekend'] = feat_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        return feat_df

    def train_and_forecast(self, prophet_df, periods=30):
        # Initialize model with seasonality components
        # Added monthly and quarterly seasonality for better performance
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.08, # Slightly increased for flexibility
            seasonality_mode='multiplicative' # Better for sales data typically
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return model, forecast

    def evaluate(self, prophet_df):
        # Split into train/test (last 30 days as test)
        train = prophet_df.iloc[:-30]
        test = prophet_df.iloc[-30:]
        
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(train)
        
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        predictions = forecast.tail(30)['yhat'].values
        actuals = test['y'].values
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        # Residuals for diagnostic plots
        residuals = actuals - predictions
        
        evaluation_results = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'test_dates': test['ds'].values,
            'actuals': actuals,
            'predictions': predictions,
            'residuals': residuals
        }
        
        return evaluation_results

if __name__ == "__main__":
    forecaster = SalesForecaster()
    category_df = forecaster.get_category_data('Electronics')
    p_df = forecaster.prepare_data(category_df)
    results = forecaster.evaluate(p_df)
    print(f"Model Evaluation (MAE): {results['mae']}")
    print(f"Model Evaluation (RMSE): {results['rmse']}")
