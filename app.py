import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from model_engine import SalesForecaster
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Sales & Demand Forecaster Pro",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f3f4f6;
    }
    
    .stMetric {
        background-color: white;
        padding: 24px !important;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
    }
    
    .stMetric:hover {
        border-color: #3b82f6;
        transition: 0.3s;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #111827;
        font-weight: 700 !important;
    }
    
    .insight-card {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
    }
    
    .insight-header {
        color: #3b82f6;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 3px solid #3b82f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🏙️ Sales & Demand Forecasting Pro")
    st.markdown("Unlocking predictive business intelligence with AI-powered time-series analysis.")
    
    # Initialize Forecaster
    try:
        forecaster = SalesForecaster()
        full_df = forecaster.df
    except FileNotFoundError:
        st.error("Historical data not found. Please run data_generator.py first.")
        return

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
        st.header("Control Panel")
        
        categories = ['All'] + sorted(full_df['Category'].unique().tolist())
        selected_category = st.selectbox("Market Segment", categories)
        forecast_days = st.slider("Forecast Horizon (Days)", 14, 180, 60)
        
        st.markdown("---")
        st.markdown("### Model Config")
        st.caption("Algorithm: Facebook Prophet")
        st.caption("Seasonality: Multiplicative")
        
        if st.button("Refresh Model Weights"):
            st.rerun()

    # Data Preparation
    df = forecaster.get_category_data(selected_category)
    p_df = forecaster.prepare_data(df)
    adv_df = forecaster.get_advanced_features(df)
    
    # --- KPI HEADER ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['Total_Sales'].sum()
    avg_order = df['Total_Sales'].mean()
    growth_val = (p_df['y'].tail(30).sum() / p_df['y'].shift(30).tail(30).sum() - 1) * 100
    
    col1.metric("Lifetime Revenue", f"${total_sales/1e6:.2f}M", delta=f"{growth_val:.1f}%")
    col2.metric("Avg Transaction", f"${avg_order:.2f}")
    col3.metric("Volume", f"{df['Units_Sold'].sum():,}")
    col4.metric("SKU Count", df['Product_ID'].nunique())
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN TABS ---
    tab_trend, tab_temp, tab_forecast, tab_diag = st.tabs([
        "📈 Sales Performance", 
        "🗓️ Temporal Analysis", 
        "🔮 Predictive Forecast", 
        "🛠️ Model Diagnostics"
    ])

    with tab_trend:
        c1, c2 = st.columns([2, 1])
        with c1:
            daily_sales = df.groupby('Date')['Total_Sales'].sum().reset_index()
            fig = px.area(daily_sales, x='Date', y='Total_Sales', 
                         title="Revenue Trajectory",
                         color_discrete_sequence=['#3b82f6'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("### Top Contributors")
            top_df = df.groupby('Product_ID')['Total_Sales'].sum().nlargest(8).reset_index()
            fig_bar = px.bar(top_df, x='Total_Sales', y='Product_ID', 
                            orientation='h', color='Total_Sales',
                            color_continuous_scale='Blues')
            fig_bar.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_temp:
        st.markdown("### Seasonality & Patterns")
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            # Heatmap of Day vs Month
            heatmap_data = adv_df.groupby(['Month', 'DayOfWeek'])['Total_Sales'].mean().unstack()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig_sns, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, ax=ax, 
                        xticklabels=days, yticklabels=months[:len(heatmap_data)])
            ax.set_title("Average Sales Intensity Heatmap")
            st.pyplot(fig_sns)
            
        with col_t2:
            # Weekend vs Weekday
            w_data = adv_df.groupby('IsWeekend')['Total_Sales'].mean().reset_index()
            w_data['Label'] = w_data['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
            fig_w = px.pie(w_data, values='Total_Sales', names='Label', 
                          title="Revenue Distribution: Weekend vs Weekday",
                          color_discrete_sequence=['#3b82f6', '#93c5fd'], hole=.4)
            st.plotly_chart(fig_w, use_container_width=True)

    with tab_forecast:
        st.markdown("### Future Outlook")
        with st.spinner("Analyzing variables and projecting trends..."):
            model, forecast = forecaster.train_and_forecast(p_df, periods=forecast_days)
            
            fig_f = go.Figure()
            # History
            fig_f.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Historical", line=dict(color="#94a3b8", width=1)))
            # Forecast
            f_part = forecast.tail(forecast_days)
            fig_f.add_trace(go.Scatter(x=f_part['ds'], y=f_part['yhat'], 
                                     name="Forecast", line=dict(color="#2563eb", width=3)))
            # Intervals
            fig_f.add_trace(go.Scatter(x=f_part['ds'], y=f_part['yhat_upper'], 
                                     fill=None, mode='lines', line_color='rgba(37, 99, 235, 0.1)', showlegend=False))
            fig_f.add_trace(go.Scatter(x=f_part['ds'], y=f_part['yhat_lower'], 
                                     fill='tonexty', mode='lines', line_color='rgba(37, 99, 235, 0.1)', name="Confidence Interval"))
            
            fig_f.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_f, use_container_width=True)

        # Insights Section
        st.markdown("---")
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            st.markdown("""
                <div class="insight-card">
                    <div class="insight-header">🔥 Growth Trajectory</div>
                    Projected demand indicators suggest a stable growth path. 
                    The model accounts for historical holiday peaks and recent momentum.
                </div>
            """, unsafe_allow_html=True)
            
            # Smart recommendation
            expected_total = f_part['yhat'].sum()
            st.success(f"💡 Recommendation: Prepare for approximately ${expected_total:,.0f} in revenue for the upcoming {forecast_days} days.")

        with c_i2:
            st.markdown("""
                <div class="insight-card">
                    <div class="insight-header">⚠️ Strategic Risks</div>
                    Watch for weekly fluctuations. The 'multiplicative' seasonality model 
                    indicates higher volatility during peak sales periods. 
                </div>
            """, unsafe_allow_html=True)

    with tab_diag:
        st.markdown("### model Reliability & Accuracy")
        eval_res = forecaster.evaluate(p_df)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (Mean Absolute Error)", f"${eval_res['mae']:,.2f}")
        m2.metric("RMSE", f"${eval_res['rmse']:,.2f}")
        rel_err = (eval_res['mae'] / p_df['y'].mean()) * 100
        m3.metric("Rel. Forecast Error", f"{rel_err:.1f}%")
        
        st.markdown("#### Residual Analysis (Actual vs Prediction Error)")
        # Plotting residuals using Matplotlib
        fig_res, ax_res = plt.subplots(figsize=(12, 4))
        ax_res.scatter(eval_res['test_dates'], eval_res['residuals'], alpha=0.5, color='#3b82f6')
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_ylabel("Error ($)")
        ax_res.set_title("Prediction Residuals over last 30 days")
        st.pyplot(fig_res)
        
        st.info("A balanced distribution around zero indicates a healthy model. Significant clusters suggest missing variables (e.g., promotions or external events).")

if __name__ == "__main__":
    main()
