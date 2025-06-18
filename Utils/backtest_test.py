import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from POTest import (
    select_top_stocks,
    allocate_portfolio,
    calculate_transaction_costs,
    format_vnd,
    calculate_performance_metrics,
    save_results
)
import warnings
warnings.filterwarnings('ignore')

PRICE_SCALE = 1000  # Nhân với 1000 để ra giá thực

def load_prediction_functions():
    """Import các hàm cần thiết từ file LSTM"""
    import pickle
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError
    from scipy.signal import savgol_filter
    
    return {
        'load_model': load_model,
        'pickle': pickle,
        'savgol_filter': savgol_filter,
        'Adam': Adam,
        'MeanSquaredError': MeanSquaredError,
        'MeanAbsoluteError': MeanAbsoluteError
    }

def predict_stocks_at_date(df, prediction_date, models_dir='Modified_Tunned_Model', 
                          predict_days=21, min_stocks=20):
    """Predict stock prices at specific date for all available models"""
    funcs = load_prediction_functions()
    
    # Chỉ lấy dữ liệu TRƯỚC ngày dự đoán (để tránh lookahead bias)
    historical_data = df[df['data_date'] < prediction_date].copy()
    
    available_tickers = []
    for ticker in historical_data['ticker'].unique():
        model_path = f"{models_dir}/{ticker}_lstm_model.h5"
        if os.path.exists(model_path):
            available_tickers.append(ticker)
    
    print(f"Tìm thấy {len(available_tickers)} mô hình khả dụng")
    
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
            print(f"Lỗi dự đoán {ticker}: {str(e)}")
            continue
    
    print(f"Dự đoán thành công {len(predictions)} cổ phiếu")
    
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def predict_single_stock(ticker, df, prediction_date, models_dir, predict_days, funcs):
    """Predict single stock using saved model"""
    from sklearn.preprocessing import MinMaxScaler
    
    model_path = f"{models_dir}/{ticker}_lstm_model.h5"
    scaler_path = f"{models_dir}/{ticker}_scaler.pkl"
    params_path = f"{models_dir}/{ticker}_params.pkl"
    
    # Kiểm tra file tồn tại
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
    # Chỉ lấy dữ liệu TRƯỚC ngày dự đoán
    ticker_data = ticker_data[ticker_data['data_date'] < prediction_date]
    
    if len(ticker_data) < 50:
        return None
    
    features = prepare_features_simple(ticker_data, funcs['savgol_filter'])
    feature_cols = params.get('feature_cols', ['sg_filtered_price', 'MA_5', 'MA_10', 'EMA_5', 'EMA_10'])
    
    # Kiểm tra và thêm cột thiếu
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
    # Fixed: Sửa lỗi cú pháp trong vòng lặp
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
    
    # Cải thiện tính toán prediction_accuracy
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

def prepare_features_simple(ticker_data, savgol_filter):
    """Simplified feature preparation"""
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

def get_price_at_date(df, ticker, date):
    """Get stock price at specific date"""
    # Lấy giá gần nhất tại hoặc trước ngày date
    ticker_data = df[(df['ticker'] == ticker) & (df['data_date'] <= date)]
    if len(ticker_data) == 0:
        return None
    return ticker_data.iloc[-1]['close_price']

def calculate_portfolio_value(df, allocation, date, shares_dict):
    """Calculate portfolio value at specific date using actual shares held"""
    total_value = 0
    successful_calculations = 0
    
    for _, row in allocation.iterrows():
        ticker = row['ticker']
        current_price = get_price_at_date(df, row['ticker'], date)
        
        if current_price is not None and ticker in shares_dict:
            stock_value = shares_dict[ticker] * current_price * PRICE_SCALE
            total_value += stock_value
            successful_calculations += 1
    
    # Chỉ trả về giá trị nếu có ít nhất 50% cổ phiếu được tính toán thành công
    if successful_calculations >= len(allocation) * 0.5:
        return total_value
    return None

def calculate_shares_from_allocation(allocation, portfolio_value):
    """Calculate actual shares to buy for each stock"""
    shares_dict = {}
    
    for _, row in allocation.iterrows():
        ticker = row['ticker']
        weight = row['weight']
        initial_price = row['initial_price']
        
        if initial_price > 0:
            allocation_amount = portfolio_value * weight
            shares = allocation_amount / (initial_price * PRICE_SCALE)
            shares_dict[ticker] = shares
    
    return shares_dict

def run_backtest_with_predictions(
    data_path,
    start_date,
    end_date,
    rebalance_frequency='monthly',
    initial_capital=100000000,  # 100 triệu VND
    k_stocks=7,
    confidence_levels=[0.90, 0.95, 0.99],
    transaction_cost_rate=0.003  # 0.3%
):
    """Run backtest with real-time predictions at each rebalance date."""
    # 1. Load & chuẩn bị dữ liệu
    df = pd.read_csv(data_path)
    df['data_date'] = pd.to_datetime(df['data_date'])
    df = df.sort_values(['ticker', 'data_date'])
    
    if 'gemini_sentiment' in df.columns:
        df['final_sentiment'] = df['gemini_sentiment'].fillna(0)
    
    # 2. Tạo danh sách ngày rebalance (đầu mỗi tháng)
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # 3. Khởi tạo kết cấu lưu kết quả
    results = {
        conf: {
            'dates': [],           
            'values': [],          
            'returns': [],         
            'allocations': [],     
            'transaction_costs': [] 
        }
        for conf in confidence_levels
    }
    
    # 4. Vòng lặp theo từng mức độ tin cậy
    for confidence in confidence_levels:
        print(f"\n{'='*60}\nBacktest với mức độ tin cậy: {confidence}\n{'='*60}")
        portfolio_value = initial_capital
        current_shares = {}  # Dictionary lưu số cổ phiếu hiện tại
        previous_weights = None
        previous_tickers = None
        
        # 5. Vòng lặp theo từng ngày rebalance
        for i, rebalance_date in enumerate(rebalance_dates, start=1):
            print(f"\n[{i}/{len(rebalance_dates)}] Tái cân bằng ngày {rebalance_date.date()}")
            
            # 5.0 Tính giá trị danh mục hiện tại (nếu có cổ phiếu từ kỳ trước)
            if current_shares:
                current_portfolio_value = 0
                for ticker, shares in current_shares.items():
                    current_price = get_price_at_date(df, ticker, rebalance_date)
                    if current_price is not None:
                        current_portfolio_value += shares * current_price * PRICE_SCALE
                
                if current_portfolio_value > 0:
                    portfolio_value = current_portfolio_value
                    print(f"Giá trị danh mục hiện tại: {format_vnd(portfolio_value)}")
            
            # 5.1 Dự đoán giá cổ phiếu
            try:
                preds = predict_stocks_at_date(df, rebalance_date)
            except Exception as e:
                print(f"Lỗi trong quá trình dự đoán: {str(e)}")
                continue
                
            if len(preds) < k_stocks:
                print(f"Cảnh báo: chỉ có {len(preds)} dự đoán", end='')
                if len(preds) < 3:
                    print("; bỏ qua kỳ này.")
                    continue
                print("; vẫn tiếp tục với số cổ phiếu hiện có.")
            
            # 5.2 Lựa chọn & phân bổ danh mục
            try:
                selected = select_top_stocks(preds, k=min(k_stocks, len(preds)))
                hist = df[df['data_date'] <= rebalance_date]
                alloc = allocate_portfolio(
                    selected, hist,
                    method='mcvar',
                    confidence_level=confidence
                )
                
                # Thêm giá khởi điểm
                alloc['initial_price'] = alloc['ticker'].apply(
                    lambda t: get_price_at_date(df, t, rebalance_date)
                )
                
                # Loại bỏ cổ phiếu không có giá
                alloc = alloc.dropna(subset=['initial_price'])
                if len(alloc) == 0:
                    print("Không có cổ phiếu hợp lệ, bỏ qua kỳ này.")
                    continue
                    
            except Exception as e:
                print(f"Lỗi trong quá trình phân bổ danh mục: {str(e)}")
                continue
            
            # 5.3 Tính chi phí giao dịch
            new_weights = alloc['weight'].values
            new_tickers = alloc['ticker'].values
            tx_cost = calculate_transaction_costs(
                portfolio_value,
                previous_weights, new_weights,
                previous_tickers, new_tickers,
                transaction_cost_rate
            )
            
            # Trừ chi phí giao dịch TRƯỚC KHI mua cổ phiếu
            portfolio_value_after_costs = max(0, portfolio_value - tx_cost)
            results[confidence]['transaction_costs'].append(tx_cost)
            print(f"Chi phí giao dịch: {format_vnd(tx_cost)}")
            print(f"Giá trị sau chi phí: {format_vnd(portfolio_value_after_costs)}")
            
            # 5.4 Tính số cổ phiếu mua cho danh mục mới
            current_shares = calculate_shares_from_allocation(alloc, portfolio_value_after_costs)
            
            # 5.5 Lưu thông tin phân bổ
            alloc_info = []
            for _, row in alloc.iterrows():
                ticker = row['ticker']
                shares = current_shares.get(ticker, 0)
                alloc_info.append({
                    'ticker': ticker,
                    'weight': float(row['weight']),
                    'expected_return': float(row.get('expected_return', 0)),
                    'price': float(row['initial_price'] * PRICE_SCALE),
                    'shares': float(shares)
                })
            
            results[confidence]['allocations'].append({
                'date': rebalance_date.strftime('%Y-%m-%d'),
                'stocks': alloc_info,
                'transaction_cost': float(tx_cost),
                'portfolio_value_before_costs': float(portfolio_value),
                'portfolio_value_after_costs': float(portfolio_value_after_costs)
            })
            
            # 5.6 Track giá trị danh mục hàng ngày đến kỳ rebalance kế tiếp
            if i < len(rebalance_dates):
                next_reb = rebalance_dates[i]
            else:
                next_reb = pd.to_datetime(end_date)
            
            tracking_dates = pd.date_range(start=rebalance_date, end=next_reb, freq='D')
            tracking_dates = tracking_dates[tracking_dates <= pd.to_datetime(end_date)]
            
            period_values = []
            for d in tracking_dates:
                val = calculate_portfolio_value(df, alloc, d, current_shares)
                if val is not None:
                    results[confidence]['dates'].append(d)
                    results[confidence]['values'].append(val)
                    period_values.append(val)
            
            # 5.7 Cập nhật portfolio_value = giá trị cuối kỳ tracking
            if period_values:
                portfolio_value = period_values[-1]
            else:
                portfolio_value = portfolio_value_after_costs
            
            # 5.8 Tính return của kỳ (so với giá trị đầu kỳ)
            initial_period_value = portfolio_value_after_costs if i == 1 else prev_portfolio_value
            if initial_period_value > 0:
                period_return = (portfolio_value / initial_period_value - 1) * 100
            else:
                period_return = 0
                
            results[confidence]['returns'].append(period_return)
            print(f"Return kỳ {i}: {period_return:.2f}% — Giá trị cuối kỳ: {format_vnd(portfolio_value)}")
            
            # 5.9 Lưu cho kỳ kế tiếp
            prev_portfolio_value = portfolio_value
            previous_weights = new_weights
            previous_tickers = new_tickers
    
    return results

def main():
    """Main backtest execution"""
    data_path = 'data/stock_sentiment_mapping.csv'
    
    # Backtest parameters
    start_date = '2023-01-01'
    end_date = '2025-06-01'
    initial_capital = 100000000  # 100 triệu VND
    confidence_levels = [0.90, 0.95, 0.99]
    transaction_cost_rate = 0.003  # 0.3%
    
    print("Bắt đầu backtest danh mục với dự đoán real-time...")
    print(f"Thời gian: {start_date} đến {end_date}")
    print(f"Vốn ban đầu: {format_vnd(initial_capital)}")
    print(f"Mức độ tin cậy: {confidence_levels}")
    print(f"Chi phí giao dịch: {transaction_cost_rate*100}%")
    
    # Kiểm tra file dữ liệu
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file dữ liệu: {data_path}")
        return
    
    # Run backtest
    try:
        results = run_backtest_with_predictions(
            data_path=data_path,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            confidence_levels=confidence_levels,
            transaction_cost_rate=transaction_cost_rate
        )
        
        # Calculate and display metrics
        metrics = calculate_performance_metrics(results)
        
        print("\n" + "="*70)
        print("KẾT QUẢ BACKTEST")
        print("="*70)
        
        for confidence, metric in metrics.items():
            print(f"\nMức độ tin cậy: {confidence}")
            print(f"Lợi nhuận tổng: {metric['total_return']:.2f}%")
            print(f"Lợi nhuận hàng năm: {metric['annual_return']:.2f}%")
            print(f"Độ biến động: {metric['volatility']:.2f}%")
            print(f"Tỷ lệ Sharpe: {metric['sharpe_ratio']:.3f}")
            print(f"Tỷ lệ Sortino: {metric['sortino_ratio']:.3f}")
            print(f"Drawdown tối đa: {metric['max_drawdown']:.2f}%")
            print(f"Giá trị cuối: {format_vnd(metric['final_value'])}")
            print(f"Tổng chi phí giao dịch: {format_vnd(metric['total_transaction_costs'])} ({metric['tx_costs_percentage']:.2f}%)")
        
        # Save results
        save_results(results, metrics)
        
        print(f"\n✅ Hoàn thành backtest!")
        print(f"📁 Kết quả đã lưu tại: backtest_results/")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình backtest: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()