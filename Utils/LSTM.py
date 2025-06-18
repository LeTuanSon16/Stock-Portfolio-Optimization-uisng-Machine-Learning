import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from scipy.signal import savgol_filter
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODELS_DIR = "Modified_Tunned_Model"
METRICS_FILE = f"{MODELS_DIR}/training_metrics.json"
PORTFOLIO_RESULTS_FILE = f"{MODELS_DIR}/portfolio_predictions.json"
os.makedirs(MODELS_DIR, exist_ok=True)

def apply_savitzky_golay_filter(data, window_length=3, polyorder=1):
    """Apply Savitzky-Golay filter for smoothing time series data"""
    if len(data) < window_length:
        return data

    if window_length % 2 == 0:
        window_length += 1

    try:
        smoothed_data = savgol_filter(data, window_length, polyorder, mode='nearest')
        return smoothed_data
    except:
        return data

def calculate_moving_averages(prices, ma_periods=[5, 10, 20]):
    """Calculate Simple Moving Average (MA) and Exponential Moving Average (EMA)"""
    ma_dict = {}
    ema_dict = {}

    for period in ma_periods:
        ma_dict[f'MA_{period}'] = prices.rolling(window=period, min_periods=1).mean()
        ema_dict[f'EMA_{period}'] = prices.ewm(span=period, adjust=False, min_periods=1).mean()

    return ma_dict, ema_dict

def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive metrics including MPA"""
    y_pred = y_pred.flatten()

    # Standard metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE
    mask = y_true != 0
    mape = 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if np.any(mask) else 0

    # MPA (Mean Prediction Accuracy) - từ paper DP-LSTM
    mpa = 1 - np.mean(np.abs(y_true - y_pred) / np.abs(y_true + 1e-8))
    mpa = max(0, min(1, mpa))  # Clamp between 0 and 1

    # Directional accuracy (price movement prediction)
    if len(y_true) > 1:
        y_true_movement = np.diff(y_true) > 0
        y_pred_movement = np.diff(y_pred) > 0
        
        min_len = min(len(y_true_movement), len(y_pred_movement))
        y_true_movement = y_true_movement[:min_len]
        y_pred_movement = y_pred_movement[:min_len]
        
        TP = np.sum((y_true_movement == 1) & (y_pred_movement == 1))
        TN = np.sum((y_true_movement == 0) & (y_pred_movement == 0))
        FP = np.sum((y_true_movement == 0) & (y_pred_movement == 1))
        FN = np.sum((y_true_movement == 1) & (y_pred_movement == 0))
        
        directional_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    else:
        TP = TN = FP = FN = 0
        directional_accuracy = 0

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'mpa': float(mpa),
        'directional_accuracy': float(directional_accuracy),
        'tp': int(TP),
        'tn': int(TN),
        'fp': int(FP),
        'fn': int(FN)
    }

def prepare_features(ticker_data, ma_periods=[5, 10, 20]):
    """Prepare features including SG filtered prices, MA, EMA, and sentiment"""
    ticker_data = ticker_data.sort_values('data_date').reset_index(drop=True)

    original_prices = ticker_data['close_price'].values
    smoothed_prices = apply_savitzky_golay_filter(original_prices)

    features_df = pd.DataFrame()
    features_df['original_price'] = original_prices
    features_df['sg_filtered_price'] = smoothed_prices

    ma_dict, ema_dict = calculate_moving_averages(pd.Series(smoothed_prices))

    for ma_name, ma_values in ma_dict.items():
        features_df[ma_name] = ma_values

    for ema_name, ema_values in ema_dict.items():
        features_df[ema_name] = ema_values

    if 'final_sentiment' in ticker_data.columns:
        features_df['sentiment'] = ticker_data['final_sentiment'].values

    if 'volume' in ticker_data.columns:
        features_df['volume'] = ticker_data['volume'].values

    return features_df

def create_sequences(data, window_size):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

def build_model(input_shape, params):
    """Build LSTM model with given parameters"""
    model = Sequential()

    if len(params['dropout_rates']) != len(params['lstm_units']):
        params['dropout_rates'] = params['dropout_rates'][:len(params['lstm_units'])]

    for i, units in enumerate(params['lstm_units']):
        return_sequences = i < len(params['lstm_units']) - 1
        if i == 0:
            model.add(LSTM(units, return_sequences=return_sequences,
                          input_shape=input_shape, activation='tanh'))
        else:
            model.add(LSTM(units, return_sequences=return_sequences, activation='tanh'))

        if i < len(params['dropout_rates']):
            model.add(Dropout(params['dropout_rates'][i]))

    for units in params['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.1))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    return model

def load_metrics():
    """Load existing metrics"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metrics(ticker, train_metrics, test_metrics, best_params):
    """Save metrics to file"""
    metrics_history = load_metrics()
    metrics_history[ticker] = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'best_params': best_params
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_history, f, indent=2)

def tune_hyperparameters(ticker, df, n_trials=10):
    """Find best hyperparameters using random search - reduced trials for efficiency"""
    param_grid = {
        'window_size': [5, 10, 15],
        'lstm_units': [[107, 79], [128, 64], [100, 50]],
        'dropout_rates': [[0.001, 0.001], [0.01, 0.01]],
        'dense_units': [[32], [64]],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [3, 8, 16],
        'epochs': [50, 75]
    }

    ticker_data = df[df['ticker'] == ticker].reset_index(drop=True)
    if len(ticker_data) < 100:
        print(f"Insufficient data for {ticker}: {len(ticker_data)} rows")
        return None, None, None

    features_df = prepare_features(ticker_data)

    feature_cols = ['sg_filtered_price', 'MA_5', 'MA_10', 'EMA_5', 'EMA_10']
    if 'sentiment' in features_df.columns:
        feature_cols.append('sentiment')
    if 'volume' in features_df.columns:
        feature_cols.append('volume')

    features_df = features_df.dropna()
    features = features_df[feature_cols].values

    split_idx = int(len(features) * 0.8)
    train_features = features[:split_idx]
    test_features = features[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    best_score = float('inf')
    best_params = None

    for trial in range(n_trials):
        params = {k: v[np.random.randint(len(v))] for k, v in param_grid.items()}

        try:
            X_train, y_train = create_sequences(train_scaled, params['window_size'])
            X_test, y_test = create_sequences(test_scaled, params['window_size'])

            if len(X_train) < 20 or len(X_test) < 5:
                continue

            model = build_model((X_train.shape[1], X_train.shape[2]), params)

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]

            history = model.fit(
                X_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )

            y_pred = model.predict(X_test, verbose=0)
            score = mean_squared_error(y_test, y_pred)

            if score < best_score:
                best_score = score
                best_params = params
                best_params['feature_cols'] = feature_cols

        except Exception as e:
            continue

    return best_params, scaler, features_df

def train_optimized_model(ticker, df, force_retrain=False):
    """Train model with hyperparameter tuning"""
    model_path = f"{MODELS_DIR}/{ticker}_lstm_model.h5"
    scaler_path = f"{MODELS_DIR}/{ticker}_scaler.pkl"
    params_path = f"{MODELS_DIR}/{ticker}_params.pkl"

    if not force_retrain and os.path.exists(model_path):
        print(f"✓ Model for {ticker} exists. Skipping...")
        return True

    print(f"Training {ticker}...")
    best_params, scaler, features_df = tune_hyperparameters(ticker, df)

    if best_params is None:
        print(f"✗ Training failed for {ticker}")
        return False

    features = features_df[best_params['feature_cols']].values

    split_idx = int(len(features) * 0.8)
    train_features = features[:split_idx]
    test_features = features[split_idx:]

    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    X_train, y_train = create_sequences(train_scaled, best_params['window_size'])
    X_test, y_test = create_sequences(test_scaled, best_params['window_size'])

    model = build_model((X_train.shape[1], X_train.shape[2]), best_params)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    # Calculate metrics
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    # Inverse transform for actual prices
    y_train_actual = scaler.inverse_transform(
        np.hstack([y_train.reshape(-1, 1),
                  np.zeros((len(y_train), len(best_params['feature_cols'])-1))])
    )[:, 0]
    y_train_pred_actual = scaler.inverse_transform(
        np.hstack([y_train_pred,
                  np.zeros((len(y_train_pred), len(best_params['feature_cols'])-1))])
    )[:, 0]

    y_test_actual = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1, 1),
                  np.zeros((len(y_test), len(best_params['feature_cols'])-1))])
    )[:, 0]
    y_test_pred_actual = scaler.inverse_transform(
        np.hstack([y_test_pred,
                  np.zeros((len(y_test_pred), len(best_params['feature_cols'])-1))])
    )[:, 0]

    train_metrics = calculate_comprehensive_metrics(y_train_actual, y_train_pred_actual)
    test_metrics = calculate_comprehensive_metrics(y_test_actual, y_test_pred_actual)

    # Save model and data
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)

    save_metrics(ticker, train_metrics, test_metrics, best_params)

    print(f"✓ {ticker} - Test: R²={test_metrics['r2']:.3f}, MPA={test_metrics['mpa']:.3f}, Acc={test_metrics['directional_accuracy']:.3f}")

    return True

def predict_monthly_prices(ticker, df, predict_days=21):
    """Predict monthly prices for portfolio optimization"""
    model_path = f"{MODELS_DIR}/{ticker}_lstm_model.h5"
    scaler_path = f"{MODELS_DIR}/{ticker}_scaler.pkl"
    params_path = f"{MODELS_DIR}/{ticker}_params.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, params_path]):
        raise FileNotFoundError(f"Missing files for {ticker}")

    # Load model with error handling
    custom_objects = {
        'mse': MeanSquaredError(),
        'MeanSquaredError': MeanSquaredError(),
        'MeanAbsoluteError': MeanAbsoluteError()
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects)
    except:
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(), loss=MeanSquiredError(), metrics=[MeanAbsoluteError()])

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    ticker_data = df[df['ticker'] == ticker].reset_index(drop=True)
    features_df = prepare_features(ticker_data)
    features_df = features_df.dropna()

    feature_cols = params.get('feature_cols', ['sg_filtered_price', 'MA_5', 'MA_10', 'EMA_5', 'EMA_10'])
    features = features_df[feature_cols].values
    scaled_features = scaler.transform(features)

    window_size = params['window_size']
    last_sequence = scaled_features[-window_size:]
    current_sequence = last_sequence.copy()

    predictions = []
    for _ in range(predict_days):
        input_seq = current_sequence.reshape(1, window_size, -1)
        next_pred = model.predict(input_seq, verbose=0)[0, 0]

        next_features = current_sequence[-1].copy()
        next_features[0] = next_pred

        current_sequence = np.vstack([current_sequence[1:], next_features])
        predictions.append(next_pred)

    # Convert predictions back to actual prices
    predictions = np.array(predictions).reshape(-1, 1)
    dummy_cols = np.zeros((len(predictions), len(feature_cols)-1))
    full_features = np.hstack([predictions, dummy_cols])
    predicted_prices = scaler.inverse_transform(full_features)[:, 0]

    # Calculate portfolio metrics
    current_price = ticker_data['close_price'].iloc[-1]
    final_price = predicted_prices[-1]
    monthly_return = (final_price - current_price) / current_price

    # Calculate historical metrics for portfolio optimization
    price_series = ticker_data['close_price']
    returns = price_series.pct_change().dropna()
    
    # Growth rate (average daily return)
    growth_rate = returns.mean()
    
    # Volatility (risk measure)
    volatility = returns.std()
    
    # Get model prediction accuracy from metrics
    metrics = load_metrics().get(ticker, {})
    prediction_accuracy = metrics.get('test_metrics', {}).get('mpa', 0)
    directional_accuracy = metrics.get('test_metrics', {}).get('directional_accuracy', 0)

    return {
        'ticker': ticker,
        'current_price': float(current_price),
        'predicted_price': float(final_price),
        'monthly_return': float(monthly_return),
        'growth_rate': float(growth_rate),
        'volatility': float(volatility),
        'prediction_accuracy': float(prediction_accuracy),
        'directional_accuracy': float(directional_accuracy),
        'predicted_prices': predicted_prices.tolist()
    }

def train_all_models(df, force_retrain=False):
    """Train models for all tickers"""
    tickers = df['ticker'].unique()
    print(f"Training models for {len(tickers)} tickers...")
    
    successful_models = []
    failed_models = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        
        try:
            if train_optimized_model(ticker, df, force_retrain):
                successful_models.append(ticker)
            else:
                failed_models.append(ticker)
        except Exception as e:
            print(f"✗ Error training {ticker}: {e}")
            failed_models.append(ticker)
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"✓ Successful: {len(successful_models)}")
    print(f"✗ Failed: {len(failed_models)}")
    
    if failed_models:
        print(f"Failed tickers: {', '.join(failed_models[:10])}")
    
    return successful_models

def generate_portfolio_predictions(df, successful_tickers):
    """Generate predictions for portfolio optimization"""
    print(f"\nGenerating predictions for {len(successful_tickers)} models...")
    
    portfolio_data = []
    failed_predictions = []
    
    for i, ticker in enumerate(successful_tickers, 1):
        try:
            print(f"[{i}/{len(successful_tickers)}] Predicting {ticker}...")
            prediction = predict_monthly_prices(ticker, df)
            portfolio_data.append(prediction)
        except Exception as e:
            print(f"✗ Prediction failed for {ticker}: {e}")
            failed_predictions.append(ticker)
    
    # Save portfolio data
    portfolio_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'successful_predictions': len(portfolio_data),
        'failed_predictions': len(failed_predictions),
        'predictions': portfolio_data
    }
    
    with open(PORTFOLIO_RESULTS_FILE, 'w') as f:
        json.dump(portfolio_results, f, indent=2)
    
    print(f"✓ Portfolio predictions saved: {len(portfolio_data)} successful")
    return portfolio_data

def get_portfolio_summary(portfolio_data):
    """Get summary for portfolio optimization"""
    df = pd.DataFrame(portfolio_data)
    
    print("\n=== PORTFOLIO OPTIMIZATION DATA ===")
    print(f"Total stocks: {len(df)}")
    
    # Sort by different criteria
    print("\n🔝 Top 10 by Monthly Return:")
    top_return = df.nlargest(10, 'monthly_return')[['ticker', 'monthly_return', 'prediction_accuracy']]
    print(top_return.to_string(index=False))
    
    print("\n📊 Top 10 by Prediction Accuracy:")
    top_accuracy = df.nlargest(10, 'prediction_accuracy')[['ticker', 'prediction_accuracy', 'monthly_return']]
    print(top_accuracy.to_string(index=False))
    
    print("\n⚖️ Top 10 by Risk-Adjusted Return (Return/Volatility):")
    df['risk_adjusted_return'] = df['monthly_return'] / (df['volatility'] + 0.001)
    top_sharpe = df.nlargest(10, 'risk_adjusted_return')[['ticker', 'monthly_return', 'volatility', 'risk_adjusted_return']]
    print(top_sharpe.to_string(index=False))
    
    print(f"\n📈 Summary Statistics:")
    print(f"Average Monthly Return: {df['monthly_return'].mean():.4f}")
    print(f"Average Prediction Accuracy (MPA): {df['prediction_accuracy'].mean():.4f}")
    print(f"Average Directional Accuracy: {df['directional_accuracy'].mean():.4f}")
    print(f"Stocks with positive returns: {len(df[df['monthly_return'] > 0])}")
    print(f"Stocks with MPA > 0.5: {len(df[df['prediction_accuracy'] > 0.5])}")
    
    return df

def main():
    """Main training and prediction pipeline"""
    data_path = '/content/drive/MyDrive/VNU IS/Thesis/stock_sentiment_mapping.csv'

    print("Loading data...")
    raw_data = pd.read_csv(data_path)

    # Prepare data
    df = raw_data.dropna(subset=['gemini_sentiment'])
    df['final_sentiment'] = df['gemini_sentiment']
    df['data_date'] = pd.to_datetime(df['data_date'])

    print(f"Data loaded: {len(df)} rows, {len(df['ticker'].unique())} unique tickers")

    # Train all models
    successful_tickers = train_all_models(df, force_retrain=True)
    
    if not successful_tickers:
        print("No successful models trained!")
        return

    # Generate portfolio predictions
    portfolio_data = generate_portfolio_predictions(df, successful_tickers)
    
    if portfolio_data:
        # Display summary for portfolio optimization
        portfolio_df = get_portfolio_summary(portfolio_data)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {PORTFOLIO_RESULTS_FILE}")
        print(f"📊 Model metrics saved to: {METRICS_FILE}")
        
        return portfolio_df
    else:
        print("No successful predictions generated!")
        return None

if __name__ == "__main__":
    result_df = main()