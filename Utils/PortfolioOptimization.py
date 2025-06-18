import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import warnings
warnings.filterwarnings('ignore')

def load_portfolio_predictions(file_path):
    """Load portfolio predictions from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['predictions'])

def calculate_var_cvar(returns, weights, confidence_level=0.95):
    """Calculate VaR and CVaR for portfolio"""
    portfolio_returns = returns.dot(weights)
    sorted_returns = np.sort(portfolio_returns)
    
    var_index = int(np.ceil((1 - confidence_level) * len(sorted_returns)))
    var = -sorted_returns[var_index-1] if var_index > 0 else 0
    cvar = -np.mean(sorted_returns[:var_index]) if var_index > 0 else 0
    
    return var, cvar

def calculate_sortino_ratio(returns, weights, risk_free_rate=0.03):
    """Calculate Sortino ratio for portfolio"""
    portfolio_returns = returns.dot(weights)
    excess_return = np.mean(portfolio_returns) * 252 - risk_free_rate
    
    # Downside deviation
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate/252]
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
    
    return excess_return / downside_deviation

def select_top_stocks(predictions_df, num_assets=7, selection_criteria='comprehensive'):
    """
    Select top stocks based on 3 criteria: potential returns, prediction accuracy, growth rate
    
    Parameters:
    -----------
    predictions_df: DataFrame with columns from prediction model
    num_assets: Number of assets to select
    selection_criteria: 'return', 'accuracy', 'growth', 'comprehensive'
    
    Returns:
    --------
    DataFrame: Top selected stocks with scores
    """
    df = predictions_df.copy()
    
    # Normalize metrics to 0-1 scale for fair comparison
    df['return_score'] = (df['monthly_return'] - df['monthly_return'].min()) / \
                        (df['monthly_return'].max() - df['monthly_return'].min() + 1e-8)
    
    df['accuracy_score'] = (df['prediction_accuracy'] - df['prediction_accuracy'].min()) / \
                          (df['prediction_accuracy'].max() - df['prediction_accuracy'].min() + 1e-8)
    
    df['growth_score'] = (df['growth_rate'] - df['growth_rate'].min()) / \
                        (df['growth_rate'].max() - df['growth_rate'].min() + 1e-8)
    
    # Selection based on criteria
    if selection_criteria == 'return':
        df['overall_score'] = df['return_score']
    elif selection_criteria == 'accuracy':
        df['overall_score'] = df['accuracy_score']
    elif selection_criteria == 'growth':
        df['overall_score'] = df['growth_score']
    else:  # comprehensive
        # Weighted combination as in the paper
        df['overall_score'] = (
            0.4 * df['return_score'] + 
            0.3 * df['accuracy_score'] + 
            0.3 * df['growth_score']
        )
    
    # Select top stocks
    top_stocks = df.nlargest(num_assets, 'overall_score')
    
    return top_stocks

def generate_mock_returns(tickers, periods=252, seed=42):
    """Generate mock historical returns for portfolio optimization"""
    np.random.seed(seed)
    
    # Generate correlated returns (more realistic)
    n_assets = len(tickers)
    
    # Create correlation matrix
    correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
    # Generate returns with correlation
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation * 0.0004,  # Daily volatility ~2%
        size=periods
    )
    
    return pd.DataFrame(returns, columns=tickers)

def monte_carlo_optimization(top_stocks, returns_data, confidence_level=0.95, 
                           n_simulations=100000, risk_free_rate=0.03):
    """
    Monte Carlo portfolio optimization using Sortino ratio
    
    Returns best portfolio based on highest Sortino ratio
    """
    n_assets = len(top_stocks)
    tickers = top_stocks['ticker'].values
    expected_returns = top_stocks['monthly_return'].values
    
    # Filter returns for selected stocks
    filtered_returns = returns_data[tickers]
    
    # Storage for results
    best_sortino = -np.inf
    best_portfolio = None
    
    print(f"Running {n_simulations:,} Monte Carlo simulations...")
    
    np.random.seed(42)
    for i in range(n_simulations):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # Calculate metrics
        portfolio_return = np.sum(weights * expected_returns)
        _, cvar = calculate_var_cvar(filtered_returns, weights, confidence_level)
        sortino_ratio = calculate_sortino_ratio(filtered_returns, weights, risk_free_rate)
        
        # Keep best portfolio
        if sortino_ratio > best_sortino:
            best_sortino = sortino_ratio
            best_portfolio = {
                'tickers': tickers.tolist(),
                'weights': weights.copy(),
                'expected_return': portfolio_return,
                'cvar': cvar,
                'sortino_ratio': sortino_ratio,
                'method': 'Monte Carlo'
            }
        
        if i % 20000 == 0 and i > 0:
            print(f"  Completed {i:,} simulations")
    
    return best_portfolio

def mcvar_model_1(top_stocks, returns_data, max_risk=0.05, confidence_level=0.95):
    """
    mCVaR Model 1: Maximize expected return subject to CVaR constraint
    """
    n_assets = len(top_stocks)
    tickers = top_stocks['ticker'].values
    expected_returns = top_stocks['monthly_return'].values
    filtered_returns = returns_data[tickers]
    
    def objective(weights):
        return -np.sum(weights * expected_returns)  # Negative for maximization
    
    def cvar_constraint(weights):
        _, cvar = calculate_var_cvar(filtered_returns, weights, confidence_level)
        return max_risk - cvar
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': cvar_constraint}  # CVaR constraint
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling
    x0 = np.ones(n_assets) / n_assets
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        portfolio_return = np.sum(weights * expected_returns)
        _, cvar = calculate_var_cvar(filtered_returns, weights, confidence_level)
        sortino_ratio = calculate_sortino_ratio(filtered_returns, weights)
        
        return {
            'tickers': tickers.tolist(),
            'weights': weights,
            'expected_return': portfolio_return,
            'cvar': cvar,
            'sortino_ratio': sortino_ratio,
            'method': 'mCVaR Model 1'
        }
    return None

def mcvar_model_2(top_stocks, returns_data, min_return=None, confidence_level=0.95):
    """
    mCVaR Model 2: Minimize CVaR subject to expected return constraint
    """
    n_assets = len(top_stocks)
    tickers = top_stocks['ticker'].values
    expected_returns = top_stocks['monthly_return'].values
    filtered_returns = returns_data[tickers]
    
    # Set minimum return if not provided
    if min_return is None:
        min_return = np.mean(expected_returns) * 0.8
    
    def objective(weights):
        _, cvar = calculate_var_cvar(filtered_returns, weights, confidence_level)
        return cvar
    
    def return_constraint(weights):
        portfolio_return = np.sum(weights * expected_returns)
        return portfolio_return - min_return
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': return_constraint}  # Return constraint
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling
    x0 = np.ones(n_assets) / n_assets
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        portfolio_return = np.sum(weights * expected_returns)
        _, cvar = calculate_var_cvar(filtered_returns, weights, confidence_level)
        sortino_ratio = calculate_sortino_ratio(filtered_returns, weights)
        
        return {
            'tickers': tickers.tolist(),
            'weights': weights,
            'expected_return': portfolio_return,
            'cvar': cvar,
            'sortino_ratio': sortino_ratio,
            'method': 'mCVaR Model 2'
        }
    return None

def equal_weight_portfolio(top_stocks, returns_data, confidence_level=0.95):
    """Equal weight benchmark portfolio (1/N)"""
    n_assets = len(top_stocks)
    tickers = top_stocks['ticker'].values
    weights = np.ones(n_assets) / n_assets
    
    portfolio_return = np.sum(weights * top_stocks['monthly_return'].values)
    filtered_returns = returns_data[tickers]
    _, cvar = calculate_var_cvar(filtered_returns, weights, confidence_level)
    sortino_ratio = calculate_sortino_ratio(filtered_returns, weights)
    
    return {
        'tickers': tickers.tolist(),
        'weights': weights,
        'expected_return': portfolio_return,
        'cvar': cvar,
        'sortino_ratio': sortino_ratio,
        'method': 'Equal Weight (1/N)'
    }

def optimize_portfolio_two_stage(predictions_df, num_assets=7, confidence_level=0.95, 
                               n_simulations=100000, selection_criteria='comprehensive',
                               max_risk=0.05, min_return_factor=0.8):
    """
    Two-stage portfolio optimization as described in the paper
    
    Stage 1: Stock selection based on 3 criteria
    Stage 2: Portfolio allocation using mCVaR models
    
    Parameters:
    -----------
    predictions_df: DataFrame from prediction model
    num_assets: Number of assets in portfolio (default 7 as in paper)
    confidence_level: Confidence level for CVaR (0.90, 0.95, 0.99)
    n_simulations: Number of Monte Carlo simulations
    selection_criteria: Stock selection method
    max_risk: Maximum risk level for Model 1
    min_return_factor: Minimum return factor for Model 2
    
    Returns:
    --------
    dict: Results from all optimization methods
    """
    print("=== TWO-STAGE PORTFOLIO OPTIMIZATION ===")
    print(f"Stage 1: Selecting top {num_assets} assets")
    print(f"Stage 2: Portfolio allocation with confidence level {confidence_level}")
    print()
    
    # Stage 1: Stock selection
    top_stocks = select_top_stocks(predictions_df, num_assets, selection_criteria)
    
    print(f"Selected top {num_assets} stocks based on {selection_criteria} criteria:")
    for i, (_, stock) in enumerate(top_stocks.iterrows()):
        print(f"  {i+1}. {stock['ticker']}: Return={stock['monthly_return']:.4f}, "
              f"Accuracy={stock['prediction_accuracy']:.3f}, Score={stock['overall_score']:.3f}")
    print()
    
    # Generate historical returns for optimization
    tickers = top_stocks['ticker'].values
    returns_data = generate_mock_returns(tickers)
    
    # Stage 2: Portfolio optimization
    results = {}
    
    # Monte Carlo optimization (baseline)
    print("Running Monte Carlo optimization...")
    mc_result = monte_carlo_optimization(top_stocks, returns_data, confidence_level, n_simulations)
    if mc_result:
        results['Monte Carlo'] = mc_result
    
    # mCVaR Model 1: Maximize return subject to risk constraint
    print("Running mCVaR Model 1 (Maximize Return)...")
    model1_result = mcvar_model_1(top_stocks, returns_data, max_risk, confidence_level)
    if model1_result:
        results['mCVaR Model 1'] = model1_result
    
    # mCVaR Model 2: Minimize risk subject to return constraint  
    print("Running mCVaR Model 2 (Minimize Risk)...")
    min_return = top_stocks['monthly_return'].mean() * min_return_factor
    model2_result = mcvar_model_2(top_stocks, returns_data, min_return, confidence_level)
    if model2_result:
        results['mCVaR Model 2'] = model2_result
    
    # Equal weight benchmark
    print("Running Equal Weight benchmark...")
    equal_result = equal_weight_portfolio(top_stocks, returns_data, confidence_level)
    results['Equal Weight'] = equal_result
    
    print(f"✓ Portfolio optimization completed with {len(results)} methods")
    
    return results, top_stocks

def get_best_portfolio(results, criterion='sortino_ratio'):
    """
    Get best portfolio based on specified criterion
    
    Parameters:
    -----------
    results: dict of portfolio results
    criterion: 'sortino_ratio', 'return_risk_ratio', 'expected_return'
    
    Returns:
    --------
    tuple: (method_name, portfolio_dict)
    """
    best_score = -np.inf
    best_method = None
    
    for method, portfolio in results.items():
        if criterion == 'sortino_ratio':
            score = portfolio.get('sortino_ratio', 0)
        elif criterion == 'return_risk_ratio':
            score = portfolio['expected_return'] / portfolio['cvar'] if portfolio['cvar'] > 0 else np.inf
        elif criterion == 'expected_return':
            score = portfolio['expected_return']
        else:
            score = portfolio.get('sortino_ratio', 0)
        
        if score > best_score:
            best_score = score
            best_method = method
    
    return best_method, results[best_method]

def print_portfolio_results(results, show_weights=True):
    """Print formatted portfolio optimization results"""
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*80)
    
    # Header
    print(f"{'Method':<20} {'Return':<10} {'CVaR':<10} {'Sortino':<10} {'Ret/Risk':<10}")
    print("-" * 80)
    
    # Results for each method
    for method, portfolio in results.items():
        return_risk_ratio = portfolio['expected_return'] / portfolio['cvar'] if portfolio['cvar'] > 0 else np.inf
        sortino = portfolio.get('sortino_ratio', 0)
        
        print(f"{method:<20} {portfolio['expected_return']:<10.4f} {portfolio['cvar']:<10.4f} "
              f"{sortino:<10.2f} {return_risk_ratio:<10.2f}")
        
        if show_weights:
            # Show asset allocation
            weights = portfolio['weights']
            tickers = portfolio['tickers']
            
            # Sort by weight (descending)
            weight_pairs = list(zip(tickers, weights))
            weight_pairs.sort(key=lambda x: x[1], reverse=True)
            
            weight_str = " | ".join([f"{ticker}:{weight:.3f}" for ticker, weight in weight_pairs])
            print(f"{'':>20} Allocation: {weight_str}")
            print()
    
    # Best portfolio summary
    best_method, best_portfolio = get_best_portfolio(results, 'sortino_ratio')
    print(f"🏆 Best Portfolio (Sortino Ratio): {best_method}")
    print(f"   Return: {best_portfolio['expected_return']:.4f}, CVaR: {best_portfolio['cvar']:.4f}")
    print("="*80)

def calculate_portfolio_performance(portfolio, actual_returns, periods=21):
    """
    Calculate portfolio performance metrics for backtesting
    
    Parameters:
    -----------
    portfolio: Portfolio dictionary with weights and tickers
    actual_returns: DataFrame with actual stock returns
    periods: Number of periods for calculation
    
    Returns:
    --------
    dict: Performance metrics
    """
    tickers = portfolio['tickers']
    weights = portfolio['weights']
    
    # Calculate portfolio returns
    portfolio_returns = actual_returns[tickers].dot(weights)
    
    # Performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252/periods) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Downside metrics
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'portfolio_returns': portfolio_returns.tolist()
    }

def save_portfolio_results(results, file_path):
    """Save portfolio optimization results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for method, portfolio in results.items():
        serializable_portfolio = portfolio.copy()
        if isinstance(serializable_portfolio['weights'], np.ndarray):
            serializable_portfolio['weights'] = serializable_portfolio['weights'].tolist()
        serializable_results[method] = serializable_portfolio
    
    with open(file_path, 'w') as f:
        json.dump({
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"Portfolio results saved to: {file_path}")

def load_portfolio_results(file_path):
    """Load portfolio optimization results from JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['results']