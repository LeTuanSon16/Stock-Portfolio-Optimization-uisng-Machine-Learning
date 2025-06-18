import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json

def calculate_returns(prices):
    """Calculate simple returns from price series"""
    return prices.pct_change().dropna()

def calculate_cvar(returns, weights, alpha=0.95):
    """Calculate Conditional Value at Risk (CVaR)"""
    portfolio_returns = returns @ weights
    var_threshold = np.percentile(portfolio_returns, (1 - alpha) * 100)
    conditional_returns = portfolio_returns[portfolio_returns <= var_threshold]
    
    if len(conditional_returns) == 0:
        return 0
    return -np.mean(conditional_returns)

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio expected return and CVaR"""
    expected_return = np.mean(returns @ weights)
    cvar_95 = calculate_cvar(returns, weights, alpha=0.95)
    return expected_return, cvar_95

def optimize_mean_cvar(returns, alpha=0.95, target_return=None):
    """
    Optimize portfolio using Mean-CVaR model
    Maximize expected return subject to CVaR constraint
    """
    n_assets = returns.shape[1]
    
    # Initial weights (equal weighted)
    init_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
    ]
    
    if target_return is not None:
        constraints.append({
            'type': 'ineq', 
            'fun': lambda w: np.mean(returns @ w) - target_return
        })
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Objective: minimize CVaR
    result = minimize(
        lambda w: calculate_cvar(returns, w, alpha),
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        return result.x
    else:
        # Fallback to equal weights
        return init_weights

def select_top_stocks(predictions_df, k=7, method='combined'):
    """Select top K stocks from predictions"""
    if len(predictions_df) == 0:
        return pd.DataFrame()
    
    # Remove stocks with extreme predictions
    predictions_df = predictions_df[
        (predictions_df['monthly_return'] > -0.3) & 
        (predictions_df['monthly_return'] < 0.5)
    ]
    
    # Rank stocks
    predictions_df['return_rank'] = predictions_df['monthly_return'].rank(ascending=False)
    predictions_df['accuracy_rank'] = predictions_df['prediction_accuracy'].rank(ascending=False)
    predictions_df['volatility_rank'] = predictions_df['volatility'].rank(ascending=True)
    
    # Combined score
    predictions_df['combined_score'] = (
        predictions_df['return_rank'] * 0.5 +
        predictions_df['accuracy_rank'] * 0.3 +
        predictions_df['volatility_rank'] * 0.2
    )
    
    # Select top K
    if method == 'return':
        selected = predictions_df.nlargest(k, 'monthly_return')
    elif method == 'accuracy':
        selected = predictions_df.nlargest(k, 'prediction_accuracy')
    else:  # combined
        selected = predictions_df.nsmallest(k, 'combined_score')
    
    return selected

def monte_carlo_portfolio_optimization(selected_stocks, historical_returns, n_simulations=100000):
    """
    Use Monte Carlo simulation to find optimal portfolio weights
    """
    n_assets = len(selected_stocks)
    best_sharpe = -np.inf
    best_weights = None
    
    # Risk-free rate (6% annually for Vietnam)
    risk_free_rate = 0.06 / 252  # Daily
    
    results = []
    
    for _ in range(n_simulations):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        
        # Calculate metrics
        portfolio_return = np.mean(historical_returns @ weights)
        portfolio_std = np.std(historical_returns @ weights)
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = historical_returns @ weights
        downside_returns = downside_returns[downside_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else portfolio_std
        
        sortino_ratio = (portfolio_return - risk_free_rate) / (downside_std + 1e-8)
        
        results.append({
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sortino': sortino_ratio,
            'cvar': calculate_cvar(historical_returns, weights)
        })
        
        if sortino_ratio > best_sharpe:
            best_sharpe = sortino_ratio
            best_weights = weights
    
    return best_weights, pd.DataFrame(results)

def allocate_portfolio(selected_stocks, historical_data, method='mcvar', confidence_level=0.95):
    """
    Allocate portfolio weights using specified method
    """
    tickers = selected_stocks['ticker'].values
    
    # Get historical returns for selected stocks
    returns_data = []
    for ticker in tickers:
        ticker_data = historical_data[historical_data['ticker'] == ticker]
        returns = ticker_data['close_price'].pct_change().dropna().values
        returns_data.append(returns[-252:])  # Last year of data
    
    # Align returns
    min_len = min(len(r) for r in returns_data)
    returns_matrix = np.column_stack([r[-min_len:] for r in returns_data])
    
    if method == 'equal':
        # Equal weighted portfolio
        weights = np.ones(len(tickers)) / len(tickers)
    elif method == 'mcvar':
        # Mean-CVaR optimization
        weights = optimize_mean_cvar(returns_matrix, alpha=confidence_level)
    elif method == 'monte_carlo':
        # Monte Carlo optimization
        weights, _ = monte_carlo_portfolio_optimization(selected_stocks, returns_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create allocation dataframe
    allocation = pd.DataFrame({
        'ticker': tickers,
        'weight': weights,
        'expected_return': selected_stocks['monthly_return'].values,
        'prediction_accuracy': selected_stocks['prediction_accuracy'].values
    })
    
    return allocation

def calculate_transaction_costs(portfolio_value, old_weights, new_weights, 
                               old_tickers, new_tickers, cost_rate=0.003):
    """
    Calculate transaction costs for rebalancing
    Transaction cost in Vietnam: ~0.3% (0.15% brokerage + 0.1% tax + other fees)
    """
    if old_weights is None or len(old_weights) == 0:
        # Initial allocation - pay cost on entire portfolio
        return portfolio_value * cost_rate
    
    # Calculate turnover
    total_turnover = 0
    
    # Stocks that are sold completely
    old_ticker_set = set(old_tickers)
    new_ticker_set = set(new_tickers)
    
    # Sell positions not in new portfolio
    for i, ticker in enumerate(old_tickers):
        if ticker not in new_ticker_set:
            total_turnover += old_weights[i]
    
    # Buy new positions
    for i, ticker in enumerate(new_tickers):
        if ticker not in old_ticker_set:
            total_turnover += new_weights[i]
        else:
            # Rebalance existing position
            old_idx = list(old_tickers).index(ticker)
            weight_change = abs(new_weights[i] - old_weights[old_idx])
            total_turnover += weight_change
    
    # Transaction cost = turnover * portfolio value * cost rate
    transaction_cost = total_turnover * portfolio_value * cost_rate
    
    return transaction_cost

def format_vnd(amount):
    """Format amount in VND with thousand separators"""
    return f"₫{amount:,.0f}"

def calculate_performance_metrics(results):
    """Calculate performance metrics for each strategy"""
    metrics = {}
    
    for confidence, data in results.items():
        if len(data['values']) < 2:
            continue
            
        values = np.array(data['values'])
        returns = np.diff(values) / values[:-1]
        
        # Annual return
        days = len(values)
        annual_return = (values[-1] / values[0]) ** (252 / days) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (risk-free rate = 6% for Vietnam)
        sharpe = (annual_return - 0.06) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = values / values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - 0.06) / downside_vol if downside_vol > 0 else 0
        
        # Transaction costs
        total_tx_costs = sum(data.get('transaction_costs', []))
        
        metrics[confidence] = {
            'annual_return': annual_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown * 100,
            'final_value': values[-1],
            'total_return': (values[-1] / values[0] - 1) * 100,
            'total_transaction_costs': total_tx_costs,
            'tx_costs_percentage': (total_tx_costs / values[0]) * 100
        }
    
    return metrics

def save_results(results, metrics, output_dir='backtest_results'):
    """Save backtest results"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(f'{output_dir}/performance_metrics.csv')
    
    # Save detailed results
    for confidence, data in results.items():
        if len(data['values']) > 0:
            df = pd.DataFrame({
                'date': data['dates'][:len(data['values'])],
                'portfolio_value': data['values']
            })
            df.to_csv(f'{output_dir}/portfolio_values_{confidence}.csv', index=False)
            
            # Save allocations
            if 'allocations' in data:
                import json
                with open(f'{output_dir}/allocations_{confidence}.json', 'w') as f:
                    json.dump(data['allocations'], f, indent=2)