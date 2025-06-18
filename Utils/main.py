# rolling_backtest.py

import pandas as pd
import numpy as np

from LSTM import train_all_models, generate_portfolio_predictions
from POTest import (
    optimize_portfolio_two_stage,
)
from backtest_test import backtest  # hàm backtest đã định nghĩa ở ví dụ trước

def rolling_backtest(
    df: pd.DataFrame,
    price_col: str = 'close_price',
    date_col: str = 'data_date',
    sentiment_col: str = 'gemini_sentiment',
    rebalance_freq: str = 'M',         # 'M' = hàng tháng
    num_assets: int = 7,
    cost_rate: float = 0.001,
    initial_capital: float = 1.0,
    **opt_kwargs
):
    """
    df: DataFrame gốc có các cột ['ticker', date_col, price_col, sentiment_col, ...]
    rebalance_freq: tần suất rebalance ('M','Q','W'…)
    opt_kwargs: tham số truyền cho optimize_portfolio_two_stage
    """
    # --- Chuẩn bị dữ liệu prices và danh sách ngày rebalance ---
    df[date_col] = pd.to_datetime(df[date_col])
    # pivot prices: index = lịch tháng cuối kỳ, columns = ticker
    prices = df.pivot(index=date_col, columns='ticker', values=price_col)
    # chọn ngày rebalance là month end có dữ liệu đầy đủ
    rebalance_dates = prices.resample(rebalance_freq).last().dropna(how='any').index

    # khởi tạo DataFrame weights
    tickers = prices.columns
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=tickers)

    # --- Rolling train–predict–optimize ---
    for date in rebalance_dates[:-1]:
        print(f"\n--- Rebalance at {date.date()} ---")
        # lấy data đến thời điểm này
        df_train = df[df[date_col] <= date].copy()
        df_train = df_train.dropna(subset=[sentiment_col])
        df_train['final_sentiment'] = df_train[sentiment_col]

        # 1) Train LSTM cho tất cả ticker
        success = train_all_models(df_train, force_retrain=False)
        if not success:
            raise RuntimeError(f"No models trained at {date}")

        # 2) Sinh prediction cho kỳ kế tiếp
        preds = generate_portfolio_predictions(df_train, success)
        pred_df = pd.DataFrame(preds)

        # 3) Tối ưu danh mục
        results, top_stocks = optimize_portfolio_two_stage(
            pred_df,
            num_assets=num_assets,
            **opt_kwargs
        )
        # Lấy phương pháp tốt nhất (theo sortino)
        best_method, best_port = max(
            results.items(),
            key=lambda kv: kv[1]['sortino_ratio']
        )
        w = pd.Series(best_port['weights'], index=best_port['tickers'])
        # gán về DataFrame weights, nếu thiếu ticker thì weight=0
        weights.loc[date, w.index] = w.values
        print(f" Chosen method: {best_method}")
        print(f" Weights:\n{w.to_string()}")

    # --- Chạy backtest trên toàn bộ lịch sử ---
    bt_dates = rebalance_dates  # prices và weights cùng index
    bt_prices = prices.reindex(bt_dates).dropna(how='any')
    bt_weights= weights.reindex(bt_dates).fillna(0.0)
    
    results_per_period, summary = backtest(
        bt_prices, bt_weights,
        cost_rate=cost_rate,
        initial_capital=initial_capital
    )

    return results_per_period, summary


if __name__ == '__main__':
    # Ví dụ chạy
    data_path = 'data/stock_sentiment_mapping.csv'
    df = pd.read_csv(data_path)
    results_df, summary = rolling_backtest(
        df,
        rebalance_freq='M',
        num_assets=7,
        cost_rate=0.005,
        initial_capital=1e6,
        confidence_level=0.95,
        n_simulations=50_000,
        selection_criteria='comprehensive',
        max_risk=0.05,
        min_return_factor=0.8
    )

    print("\n=== Backtest per period ===")
    print(results_df)

    print("\n=== Summary ===")
    print(summary)
    # bạn có thể lưu:
    results_df.to_csv('backtest_periods.csv')
    summary.to_frame('value').to_csv('backtest_summary.csv')
