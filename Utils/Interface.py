import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from POTest import select_top_stocks, allocate_portfolio, format_vnd
import warnings
warnings.filterwarnings('ignore')

PRICE_SCALE = 1000

# Import prediction functions from backtest
def load_prediction_functions():
    """Import prediction utilities with TensorFlow optimizations"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    import pickle
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError
    from scipy.signal import savgol_filter
    
    return {
        'load_model': load_model, 'pickle': pickle, 'savgol_filter': savgol_filter,
        'Adam': Adam, 'MeanSquaredError': MeanSquaredError, 'MeanAbsoluteError': MeanAbsoluteError
    }

def prepare_features_simple(ticker_data, savgol_filter):
    """Simplified feature preparation from backtest"""
    df = pd.DataFrame()
    
    prices = ticker_data['close_price'].values
    if len(prices) >= 3:
        sg_prices = savgol_filter(prices, 3, 1, mode='nearest')
    else:
        sg_prices = prices
    
    df['sg_filtered_price'] = sg_prices
    
    for period in [5, 10, 20]:
        df[f'MA_{period}'] = pd.Series(sg_prices).rolling(window=period, min_periods=1).mean()
        df[f'EMA_{period}'] = pd.Series(sg_prices).ewm(span=period, adjust=False, min_periods=1).mean()
    
    return df

def predict_single_stock(ticker, df, prediction_date, models_dir, predict_days, funcs):
    """Predict single stock using saved model (from backtest)"""
    model_path = f"{models_dir}/{ticker}_lstm_model.h5"
    scaler_path = f"{models_dir}/{ticker}_scaler.pkl"
    params_path = f"{models_dir}/{ticker}_params.pkl"
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [model_path, scaler_path, params_path]):
        return None
    
    custom_objects = {
        'mse': funcs['MeanSquaredError'](),
        'MeanSquaredError': funcs['MeanSquaredError'](),
        'MeanAbsoluteError': funcs['MeanAbsoluteError']()
    }
    
    try:
        model = funcs['load_model'](model_path, custom_objects=custom_objects)
    except:
        model = funcs['load_model'](model_path, compile=False)
        model.compile(optimizer=funcs['Adam'](), 
                     loss=funcs['MeanSquaredError'](), 
                     metrics=[funcs['MeanAbsoluteError']()])
    
    with open(scaler_path, 'rb') as f:
        scaler = funcs['pickle'].load(f)
    with open(params_path, 'rb') as f:
        params = funcs['pickle'].load(f)
    
    ticker_data = df[df['ticker'] == ticker].copy()
    # Only use data BEFORE prediction date
    ticker_data = ticker_data[ticker_data['data_date'] < prediction_date]
    
    if len(ticker_data) < 50:
        return None
    
    features = prepare_features_simple(ticker_data, funcs['savgol_filter'])
    feature_cols = params.get('feature_cols', ['sg_filtered_price', 'MA_5', 'MA_10', 'EMA_5', 'EMA_10'])
    
    # Check and add missing columns
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    
    features_array = features[feature_cols].values
    scaled_features = scaler.transform(features_array)
    
    window_size = params['window_size']
    if len(scaled_features) < window_size:
        return None
        
    last_sequence = scaled_features[-window_size:]
    current_sequence = last_sequence.copy()
    
    predictions = []
    for day in range(predict_days):
        input_seq = current_sequence.reshape(1, window_size, -1)
        next_pred = model.predict(input_seq, verbose=0)[0, 0]
        
        next_features = current_sequence[-1].copy()
        next_features[0] = next_pred
        
        current_sequence = np.vstack([current_sequence[1:], next_features])
        predictions.append(next_pred)
    
    predictions = np.array(predictions).reshape(-1, 1)
    dummy_cols = np.zeros((len(predictions), len(feature_cols)-1))
    full_features = np.hstack([predictions, dummy_cols])
    predicted_prices = scaler.inverse_transform(full_features)[:, 0]
    
    current_price = ticker_data['close_price'].iloc[-1]
    final_price = predicted_prices[-1]
    monthly_return = (final_price - current_price) / current_price
    
    returns = ticker_data['close_price'].pct_change().dropna()
    growth_rate = returns.mean() if len(returns) > 0 else 0
    volatility = returns.std() if len(returns) > 0 else 0.1
    
    # Improved prediction_accuracy calculation
    prediction_accuracy = max(0.5, min(0.95, 1 - volatility * 5))
    
    return {
        'ticker': ticker,
        'current_price': float(current_price),
        'predicted_price': float(final_price),
        'monthly_return': float(monthly_return),
        'growth_rate': float(growth_rate),
        'volatility': float(volatility),
        'prediction_accuracy': float(prediction_accuracy)
    }

def predict_stocks_at_date(df, prediction_date, models_dir='Modified_Tunned_Model', 
                          predict_days=21, min_stocks=20):
    """Predict stock prices at specific date for all available models (from backtest)"""
    funcs = load_prediction_functions()
    
    # Only take data BEFORE prediction date (to avoid lookahead bias)
    historical_data = df[df['data_date'] < prediction_date].copy()
    
    available_tickers = []
    for ticker in historical_data['ticker'].unique():
        model_path = f"{models_dir}/{ticker}_lstm_model.h5"
        if os.path.exists(model_path):
            available_tickers.append(ticker)
    
    predictions = []
    
    for ticker in available_tickers:
        try:
            prediction = predict_single_stock(
                ticker, historical_data, prediction_date, 
                models_dir, predict_days, funcs
            )
            if prediction is not None:
                predictions.append(prediction)
                
        except Exception as e:
            continue
    
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

@st.cache_data
def load_data():
    """Load stock data"""
    data_path = 'data/stock_sentiment_mapping.csv'
    if not os.path.exists(data_path):
        st.error(f"Không tìm thấy file dữ liệu: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['data_date'] = pd.to_datetime(df['data_date'])
    df = df.sort_values(['ticker', 'data_date'])
    return df

def load_backtest_results():
    """Load existing backtest results from CSV and JSON files"""
    results_dir = 'backtest_results'
    
    if not os.path.exists(results_dir):
        return None, None
    
    try:
        # Load performance metrics from CSV
        metrics_file = os.path.join(results_dir, 'performance_metrics.csv')
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            metrics = {}
            for _, row in metrics_df.iterrows():
                conf_level = row.iloc[0]
                metrics[str(conf_level)] = {
                    'annual_return': row['annual_return'],
                    'volatility': row['volatility'],
                    'sharpe_ratio': row['sharpe_ratio'],
                    'sortino_ratio': row['sortino_ratio'],
                    'max_drawdown': row['max_drawdown'],
                    'final_value': row['final_value'],
                    'total_return': row['total_return'],
                    'total_transaction_costs': row['total_transaction_costs'],
                    'tx_costs_percentage': row['tx_costs_percentage']
                }
        else:
            return None, None
        
        # Load portfolio values and allocations
        results = {}
        for conf_level in metrics.keys():
            portfolio_file = os.path.join(results_dir, f'portfolio_values_{conf_level}.csv')
            allocations_file = os.path.join(results_dir, f'allocations_{conf_level}.json')
            
            if os.path.exists(portfolio_file):
                portfolio_df = pd.read_csv(portfolio_file)
                dates = pd.to_datetime(portfolio_df['date'])
                values = portfolio_df['portfolio_value'].tolist()
            else:
                dates, values = [], []
            
            allocations = []
            if os.path.exists(allocations_file):
                with open(allocations_file, 'r') as f:
                    allocations = json.load(f)
            
            results[conf_level] = {
                'dates': dates,
                'values': values,
                'allocations': allocations
            }
        
        return results, metrics
        
    except Exception as e:
        st.error(f"Lỗi khi load dữ liệu: {str(e)}")
        return None, None

def get_price_at_date(df, ticker, date):
    """Get stock price at specific date (from backtest)"""
    ticker_data = df[(df['ticker'] == ticker) & (df['data_date'] <= date)]
    if len(ticker_data) == 0:
        return None
    return ticker_data.iloc[-1]['close_price']

def predict_portfolio_for_next_month(df, k_stocks=7, confidence_level=0.95, models_dir='Modified_Tunned_Model'):
    """Generate portfolio prediction for 1 month after latest data date using backtest functions"""
    
    # Get latest date in data + 1 month
    latest_date = df['data_date'].max()
    prediction_date = latest_date + timedelta(days=30)
    
    st.info(f"📅 Dữ liệu mới nhất: {latest_date.strftime('%Y-%m-%d')}")
    st.info(f"🔮 Dự đoán cho: {prediction_date.strftime('%Y-%m-%d')} (1 tháng sau)")
    
    # Use backtest prediction function
    try:
        predictions_df = predict_stocks_at_date(df, prediction_date, models_dir, predict_days=21)
        
        if len(predictions_df) == 0:
            st.error("Không có dự đoán nào thành công")
            return None
        
        st.success(f"✅ Dự đoán thành công {len(predictions_df)} cổ phiếu")
        
        # Portfolio optimization using existing functions
        try:
            selected = select_top_stocks(predictions_df, k=min(k_stocks, len(predictions_df)))
            
            # Use historical data for allocation
            hist_data = df[df['data_date'] <= latest_date]
            allocation = allocate_portfolio(selected, hist_data, method='mcvar', confidence_level=confidence_level)
            
            # Add current prices for allocation
            allocation['initial_price'] = allocation['ticker'].apply(
                lambda t: get_price_at_date(df, t, latest_date)
            )
            
            # Remove stocks without prices
            allocation = allocation.dropna(subset=['initial_price'])
            
            return {
                'predictions': predictions_df,
                'allocation': allocation if len(allocation) > 0 else None,
                'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                'latest_data_date': latest_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            st.warning(f"Lỗi tối ưu hóa: {str(e)}")
            return {
                'predictions': predictions_df,
                'allocation': None,
                'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                'latest_data_date': latest_date.strftime('%Y-%m-%d')
            }
            
    except Exception as e:
        st.error(f"Lỗi dự đoán: {str(e)}")
        return None

def show_backtest_results():
    """Display backtest results from saved files"""
    st.header("📈 Kết quả Backtest")
    
    results, metrics = load_backtest_results()
    
    if results is None or metrics is None:
        st.error("❌ Không tìm thấy kết quả backtest")
        st.info("Chạy file backtest để tạo kết quả trong thư mục backtest_results/")
        return
    
    confidence_levels = [float(k) for k in metrics.keys()]
    confidence_levels.sort()
    
    # Summary metrics
    st.subheader("📊 Tổng quan hiệu suất")
    
    cols = st.columns(len(confidence_levels))
    for i, conf in enumerate(confidence_levels):
        conf_str = str(conf)
        metric = metrics[conf_str]
        
        with cols[i]:
            st.metric(
                f"Confidence {conf:.0%}",
                f"{metric['total_return']:.2f}%",
                delta=f"Sharpe: {metric['sharpe_ratio']:.3f}"
            )
    
    # Detailed comparison table
    st.subheader("So sánh chi tiết")
    
    comparison_data = []
    for conf in confidence_levels:
        conf_str = str(conf)
        metric = metrics[conf_str]
        comparison_data.append({
            'Confidence Level': f"{conf:.0%}",
            'Total Return (%)': f"{metric['total_return']:.2f}",
            'Annual Return (%)': f"{metric['annual_return']:.2f}",
            'Volatility (%)': f"{metric['volatility']:.2f}",
            'Sharpe Ratio': f"{metric['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{metric['max_drawdown']:.2f}",
            'Final Value (VND)': f"{metric['final_value']:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Performance chart
    selected_conf = st.selectbox(
        "Chọn mức độ tin cậy:",
        confidence_levels,
        format_func=lambda x: f"{x:.0%}",
        index=1 if len(confidence_levels) > 1 else 0
    )
    
    conf_str = str(selected_conf)
    
    if conf_str in results and len(results[conf_str]['dates']) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[conf_str]['dates'],
            y=results[conf_str]['values'],
            mode='lines',
            name=f'Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(
            y=100000000,
            line_dash="dash",
            line_color="red",
            annotation_text="Vốn ban đầu: 100M VND"
        )
        
        fig.update_layout(
            title=f"Hiệu suất danh mục - Confidence {selected_conf:.0%}",
            xaxis_title="Ngày",
            yaxis_title="Giá trị (VND)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_current_prediction():
    """Display portfolio prediction for next month using backtest functions"""
    st.header("🔮 Dự đoán Danh mục")
    
    df = load_data()
    if df is None:
        return
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        k_stocks = st.slider("Số lượng cổ phiếu", 3, 15, 7)
    with col2:
        confidence_level = st.selectbox("Mức độ tin cậy", [0.90, 0.95, 0.99], index=1)
    
    if st.button("Tạo dự đoán", type="primary"):
        with st.spinner("Đang dự đoán..."):
            result = predict_portfolio_for_next_month(df, k_stocks, confidence_level)
            
            if result is None:
                return
            
            predictions = result['predictions']
            allocation = result.get('allocation')
            
            # Summary
            st.subheader("Tóm tắt dự đoán")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Số cổ phiếu", len(predictions))
            with col2:
                avg_return = predictions['monthly_return'].mean() * 100
                st.metric("Return TB (%)", f"{avg_return:.2f}")
            with col3:
                avg_accuracy = predictions['prediction_accuracy'].mean() * 100
                st.metric("Độ chính xác TB (%)", f"{avg_accuracy:.1f}")
            

            # Portfolio recommendation
            if allocation is not None and len(allocation) > 0:
                st.subheader("Danh mục đề xuất")
                
                allocation_display = allocation.copy()
                allocation_display['weight_percent'] = allocation_display['weight'] * 100
                allocation_display['expected_return_percent'] = allocation_display['expected_return'] * 100
                allocation_display['current_price_vnd'] = allocation_display['initial_price'] * PRICE_SCALE
                
                st.dataframe(
                    allocation_display[['ticker', 'weight_percent', 'expected_return_percent', 'current_price_vnd']].round(2),
                    column_config={
                        'ticker': 'Mã CK',
                        'weight_percent': st.column_config.NumberColumn('Tỷ trọng (%)', format="%.2f%%"),
                        'expected_return_percent': st.column_config.NumberColumn('Expected Return (%)', format="%.2f%%"),
                        'current_price_vnd': st.column_config.NumberColumn('Giá hiện tại (VND)', format="%,.0f")
                    },
                    use_container_width=True
                )
                
                # Pie chart
                fig_pie = px.pie(
                    allocation_display,
                    values='weight_percent',
                    names='ticker',
                    title="Cơ cấu danh mục đề xuất"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Investment calculator
                st.subheader("Tính toán đầu tư")
                
                investment_amount = st.number_input(
                    "Số tiền đầu tư (VND)",
                    min_value=1000000,
                    max_value=10000000000,
                    value=100000000,
                    step=10000000
                )
                
                st.write("**Phân bổ cụ thể:**")
                total_shares = 0
                for _, row in allocation_display.iterrows():
                    amount = investment_amount * row['weight']
                    shares = amount / row['current_price_vnd']
                    total_shares += shares
                    st.write(f"• **{row['ticker']}**: {format_vnd(amount)} ({row['weight_percent']:.1f}%) - {shares:.0f} cổ phiếu")
                
                st.info(f"Tổng cộng: {total_shares:.0f} cổ phiếu từ {len(allocation_display)} mã")
            
            else:
                st.warning("Không thể tối ưu hóa danh mục")
    
    else:
        # Show data info
        st.info("👈 Nhấn 'Tạo dự đoán' để bắt đầu")
        
        latest_date = df['data_date'].max()
        prediction_date = latest_date + timedelta(days=30)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số cổ phiếu", df['ticker'].nunique())
        with col2:
            st.metric("Dữ liệu mới nhất", latest_date.strftime('%Y-%m-%d'))
        with col3:
            st.metric("Sẽ dự đoán cho", prediction_date.strftime('%Y-%m-%d'))

def main():
    st.set_page_config(page_title="Portfolio Analysis", layout="wide")
    
    st.title("📊 Portfolio Analysis & Prediction")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Kết quả Backtest", "Dự đoán Tháng Tới"])
    
    with tab1:
        show_backtest_results()
    
    with tab2:
        show_current_prediction()

if __name__ == "__main__":
    main()