import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model_engine import SalesForecaster
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Sales & Demand Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
    }
    .insight-card {
        padding: 15px;
        border-left: 5px solid #3b82f6;
        background-color: #eff6ff;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Logic
def main():
    st.title("🚀 Sales & Demand Forecasting Dashboard")
    st.markdown("Analyze historical performance and predict future trends using AI-powered forecasting.")
    
    # Initialize Forecaster
    try:
        forecaster = SalesForecaster()
        full_df = forecaster.df
    except FileNotFoundError:
        st.error("Historical data not found. Please run data_generator.py first.")
        return

    # Sidebar Filters
    st.sidebar.header("Navigation & Filters")
    categories = ['All'] + sorted(full_df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Product Category", categories)
    
    forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 90, 30)
    
    # Filter Data
    df = forecaster.get_category_data(selected_category)
    p_df = forecaster.prepare_data(df)
    
    # --- TOP KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['Total_Sales'].sum()
    avg_order_value = df['Total_Sales'].mean()
    total_units = df['Units_Sold'].sum()
    unique_products = df['Product_ID'].nunique()
    
    col1.metric("Total Revenue", f"${total_sales:,.0f}")
    col2.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    col3.metric("Total Units Sold", f"{total_units:,}")
    col4.metric("Active Products", unique_products)
    
    st.markdown("---")
    
    # --- HISTORICAL ANALYSIS ---
    st.subheader("📊 Historical Sales Trends")
    tab1, tab2 = st.tabs(["Sales Trend", "Product Performance"])
    
    with tab1:
        fig_hist = px.line(df.groupby('Date')['Total_Sales'].sum().reset_index(), 
                         x='Date', y='Total_Sales', 
                         title=f"Daily Revenue: {selected_category}",
                         color_discrete_sequence=['#3b82f6'])
        fig_hist.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with tab2:
        top_prods = df.groupby('Product_ID')['Total_Sales'].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top_prods, x='Product_ID', y='Total_Sales', 
                        title="Top 10 Products by Revenue",
                        color='Total_Sales', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- FORECASTING SECTION ---
    st.markdown("---")
    st.subheader("🔮 AI Demand Forecast")
    
    with st.spinner("Generating forecast..."):
        model, forecast = forecaster.train_and_forecast(p_df, periods=forecast_days)
        
        # evaluation
        mae, rmse = forecaster.evaluate(p_df)
        
        # Forecast Plot
        fig_forecast = go.Figure()
        
        # Historical
        fig_forecast.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Actual Sales", line=dict(color="#64748b")))
        
        # Future Forecast
        future_forecast = forecast.tail(forecast_days)
        fig_forecast.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], 
                                        name="Predicted Sales", line=dict(color="#3b82f6", dash='dot')))
        
        # Confidence Intervals
        fig_forecast.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], 
                                        fill=None, mode='lines', line_color='rgba(59, 130, 246, 0.2)', showlegend=False))
        fig_forecast.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], 
                                        fill='tonexty', mode='lines', line_color='rgba(59, 130, 246, 0.2)', name="Confidence Range"))
        
        fig_forecast.update_layout(title="Future Sales Outlook with Prediction Intervals", 
                                  xaxis_title="Date", yaxis_title="Revenue ($)",
                                  hovermode="x unified")
        st.plotly_chart(fig_forecast, use_container_width=True)

    # --- BUSINESS INSIGHTS ---
    st.markdown("---")
    st.subheader("🧠 Business Insights & Recommendations")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.write("### 🔥 Demand Patterns")
        
        # Calculate growth
        last_month = p_df[p_df['ds'] > p_df['ds'].max() - timedelta(days=30)]['y'].sum()
        prev_month = p_df[(p_df['ds'] <= p_df['ds'].max() - timedelta(days=30)) & 
                          (p_df['ds'] > p_df['ds'].max() - timedelta(days=60))]['y'].sum()
        growth = ((last_month - prev_month) / prev_month) * 100 if prev_month > 0 else 0
        
        st.markdown(f"""
        <div class="insight-card">
        <strong>Trend Strength:</strong> Sales have shown a <strong>{growth:.1f}%</strong> 
        {'increase' if growth > 0 else 'decrease'} in the last 30 days compared to the previous month.
        </div>
        """, unsafe_allow_html=True)
        
        # Seasonal peak insight
        peak_month = df.groupby(df['Date'].dt.month)['Total_Sales'].mean().idxmax()
        month_name = datetime(2000, peak_month, 1).strftime('%B')
        
        st.markdown(f"""
        <div class="insight-card">
        <strong>Peak Season:</strong> Data suggests <strong>{month_name}</strong> is your strongest performance month historically. 
        Prepare inventory 2 weeks prior.
        </div>
        """, unsafe_allow_html=True)

    with col_in2:
        st.write("### ⚠️ Actionable Alerts")
        
        # Forecast drop alert
        next_week_avg = future_forecast.head(7)['yhat'].mean()
        curr_week_avg = p_df.tail(7)['y'].mean()
        
        if next_week_avg < curr_week_avg * 0.9:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: #ef4444; background-color: #fef2f2;">
            <strong>LOW DEMAND ALERT:</strong> Forecast predicts a 10%+ dip in sales next week. 
            Consider launching a flash sale or promotional campaign.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color: #10b981; background-color: #f0fdf4;">
            <strong>STABLE OUTLOOK:</strong> Sales are expected to remain steady. 
            Maintain current stock levels.
            </div>
            """, unsafe_allow_html=True)

        st.info(f"Model Accuracy (MAE): ${mae:,}. The forecast has a relative error of {round((mae/avg_order_value)*100, 1)}%.")

if __name__ == "__main__":
    main()
