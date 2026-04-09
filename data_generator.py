import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(days=1095): # 3 years
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    date_list = [start_date + timedelta(days=x) for x in range(days)]
    
    products = {
        'Electronics': {'base_price': 800, 'volatility': 0.2, 'growth': 0.0005},
        'Home Appliances': {'base_price': 400, 'volatility': 0.15, 'growth': 0.0003},
        'Clothing': {'base_price': 50, 'volatility': 0.3, 'growth': 0.0002},
        'Accessories': {'base_price': 25, 'volatility': 0.1, 'growth': 0.0001}
    }
    
    data = []
    
    for date in date_list:
        for product, config in products.items():
            # Seasonality: higher in Nov/Dec, end of month
            month_effect = 1.5 if date.month in [11, 12] else 1.0
            day_of_week_effect = 1.3 if date.weekday() >= 5 else 1.0
            
            # Trend
            days_passed = (date - start_date).days
            trend = 1 + (config['growth'] * days_passed)
            
            # Random noise
            noise = np.random.normal(1, config['volatility'])
            
            # Demand calculation
            base_demand = np.random.randint(5, 20)
            units_sold = int(base_demand * month_effect * day_of_week_effect * trend * noise)
            units_sold = max(0, units_sold)
            
            price = config['base_price'] * (1 + np.random.uniform(-0.05, 0.05))
            total_sales = units_sold * price
            
            data.append([date, product, f"PROD_{product[:3].upper()}_{np.random.randint(100, 999)}", units_sold, round(price, 2), round(total_sales, 2)])
            
    df = pd.DataFrame(data, columns=['Date', 'Category', 'Product_ID', 'Units_Sold', 'Price', 'Total_Sales'])
    return df

if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv('historical_sales.csv', index=False)
    print(f"Generated {len(df)} rows of sales data.")
