import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory for plots
if not os.path.exists('output_plots'):
    os.makedirs('output_plots')

def get_stock_data(symbol, start_date, end_date):
    """
    Download stock data from Yahoo Finance with additional market data
    """
    stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Handle multi-level columns if they exist
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    
    # Download market indices for correlation
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Close']
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
    dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)['Close']  # Dollar Index
    
    # Add market data
    stock['SPY'] = spy
    stock['VIX'] = vix
    stock['DXY'] = dxy
    
    return stock

def create_optimized_features(df):
    """
    Create features optimized for Lasso regression
    """
    df = df.copy()
    
    # Price-based features
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages and ratios
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'Price_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}']
        df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
        df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']
    
    # Exponential Moving Averages
    for span in [12, 26]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI with multiple periods
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [10, 20]:
        rolling_mean = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        df[f'BB_Upper_{period}'] = rolling_mean + (rolling_std * 2)
        df[f'BB_Lower_{period}'] = rolling_mean - (rolling_std * 2)
        df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
        df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / df[f'BB_Width_{period}']
    
    # Volatility features
    df['ATR_14'] = calculate_atr(df, 14)
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    df['Volatility_60'] = df['Return'].rolling(window=60).std()
    
    # Price patterns
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Support and Resistance
    for period in [10, 20, 50]:
        df[f'Resistance_{period}'] = df['High'].rolling(window=period).max()
        df[f'Support_{period}'] = df['Low'].rolling(window=period).min()
        df[f'SR_Position_{period}'] = (df['Close'] - df[f'Support_{period}']) / (df[f'Resistance_{period}'] - df[f'Support_{period}'])
    
    # Market correlation features
    if 'SPY' in df.columns:
        for period in [10, 20, 60]:
            df[f'SPY_Corr_{period}'] = df['Return'].rolling(window=period).corr(df['SPY'].pct_change())
    
    if 'VIX' in df.columns:
        df['VIX_20MA'] = df['VIX'].rolling(window=20).mean()
        df['VIX_Ratio'] = df['VIX'] / df['VIX_20MA']
    
    if 'DXY' in df.columns:
        df['DXY_Return'] = df['DXY'].pct_change()
        df['DXY_20MA'] = df['DXY'].rolling(window=20).mean()
    
    # Lag features (important for time series)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
        df[f'Volume_Lag_{lag}'] = (df['Volume'] / df['Volume_SMA_10']).shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'Return_Mean_{window}'] = df['Return'].rolling(window=window).mean()
        df[f'Return_Std_{window}'] = df['Return'].rolling(window=window).std()
        df[f'Return_Skew_{window}'] = df['Return'].rolling(window=window).skew()
    
    # Time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Month'] = df.index.month
    
    # One-hot encode day of week
    for day in range(5):  # Monday = 0, Friday = 4
        df[f'Is_Day_{day}'] = (df['Day_of_Week'] == day).astype(int)
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def optimize_lasso_alpha(X_train, y_train, X_test, y_test):
    """
    Find optimal alpha parameter for Lasso regression
    """
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    results = []
    
    print("\nOptimizing Lasso alpha parameter...")
    print("-" * 60)
    
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=2000)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Test performance
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        # Count non-zero coefficients
        non_zero = np.sum(model.coef_ != 0)
        
        results.append({
            'alpha': alpha,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_mae': test_mae,
            'non_zero_features': non_zero,
            'model': model
        })
        
        print(f"Alpha: {alpha:>6.4f} | CV R²: {cv_scores.mean():>6.3f} (+/-{cv_scores.std():>5.3f}) | "
              f"Test R²: {test_r2:>6.3f} | MAE: ${test_mae:>5.2f} | Features: {non_zero:>3d}")
    
    # Find best alpha based on CV score
    best_result = max(results, key=lambda x: x['cv_r2_mean'])
    print(f"\nBest alpha: {best_result['alpha']} with CV R² of {best_result['cv_r2_mean']:.4f}")
    
    return best_result, results

def analyze_lasso_features(model, feature_names):
    """
    Analyze which features Lasso selected
    """
    # Get coefficients
    coefs = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coef': np.abs(model.coef_)
    }).sort_values('abs_coef', ascending=False)
    
    # Separate positive and negative coefficients
    positive_coefs = coefs[coefs['coefficient'] > 0].head(10)
    negative_coefs = coefs[coefs['coefficient'] < 0].head(10)
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    print("\nTop 10 Positive Predictors (higher values → higher future price):")
    print("-" * 50)
    for idx, row in positive_coefs.iterrows():
        print(f"{row['feature']:<30} | Coef: {row['coefficient']:>8.4f}")
    
    print("\nTop 10 Negative Predictors (higher values → lower future price):")
    print("-" * 50)
    for idx, row in negative_coefs.iterrows():
        print(f"{row['feature']:<30} | Coef: {row['coefficient']:>8.4f}")
    
    # Count zero coefficients
    zero_coefs = (coefs['coefficient'] == 0).sum()
    print(f"\nFeature Selection Summary:")
    print(f"- Total features: {len(feature_names)}")
    print(f"- Features used: {len(feature_names) - zero_coefs}")
    print(f"- Features ignored: {zero_coefs}")
    
    return coefs

def analyze_prediction_errors(y_test, predictions, dates, symbol):
    """
    Detailed error analysis
    """
    errors = y_test.values - predictions
    abs_errors = np.abs(errors)
    pct_errors = (errors / y_test.values) * 100
    
    print("\n" + "="*60)
    print("PREDICTION ERROR ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\nError Statistics:")
    print(f"- Mean Error: ${np.mean(errors):.2f}")
    print(f"- Mean Absolute Error: ${np.mean(abs_errors):.2f}")
    print(f"- Error Std Dev: ${np.std(errors):.2f}")
    print(f"- Mean % Error: {np.mean(np.abs(pct_errors)):.2f}%")
    
    # Error percentiles
    print(f"\nError Percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"- {p}th percentile: ${np.percentile(abs_errors, p):.2f}")
    
    # Worst predictions
    worst_indices = np.argsort(abs_errors)[-10:][::-1]
    print(f"\nTop 10 Worst Predictions:")
    print("-" * 60)
    print(f"{'Date':<12} | {'Actual':<8} | {'Predicted':<8} | {'Error':<8} | {'% Error':<8}")
    print("-" * 60)
    
    for idx in worst_indices:
        date = dates[idx].strftime('%Y-%m-%d')
        actual = y_test.values[idx]
        pred = predictions[idx]
        error = errors[idx]
        pct_err = pct_errors[idx]
        print(f"{date:<12} | ${actual:<7.2f} | ${pred:<7.2f} | ${error:<7.2f} | {pct_err:<7.2f}%")
    
    # Plot error distribution
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Error distribution
    plt.subplot(2, 3, 1)
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error ($)')
    plt.ylabel('Frequency')
    
    # Subplot 2: Absolute error over time
    plt.subplot(2, 3, 2)
    plt.plot(dates, abs_errors, alpha=0.6)
    plt.plot(dates, pd.Series(abs_errors).rolling(20).mean(), color='red', label='20-day MA')
    plt.title('Absolute Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Absolute Error ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Subplot 3: Actual vs Predicted
    plt.subplot(2, 3, 3)
    plt.scatter(y_test.values, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    
    # Subplot 4: Residual plot
    plt.subplot(2, 3, 4)
    plt.scatter(predictions, errors, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residual ($)')
    
    # Subplot 5: Q-Q plot
    plt.subplot(2, 3, 5)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Check for Normal Distribution)')
    
    # Subplot 6: Error by day of week
    plt.subplot(2, 3, 6)
    dow_errors = pd.DataFrame({
        'error': abs_errors,
        'dow': dates.dayofweek
    })
    dow_errors.boxplot(by='dow', ax=plt.gca())
    plt.title('Error by Day of Week')
    plt.xlabel('Day (0=Monday, 4=Friday)')
    plt.ylabel('Absolute Error ($)')
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    filename = f'output_plots/{symbol}_error_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved error analysis plot: {filename}")
    plt.close()
    
    return errors, abs_errors

def create_ensemble_predictions(models_results, y_test):
    """
    Create weighted ensemble from best performing models
    """
    # Select top 3 models by CV score
    sorted_models = sorted(models_results, key=lambda x: x['cv_r2_mean'], reverse=True)[:3]
    
    # Calculate weights based on CV scores
    cv_scores = [m['cv_r2_mean'] for m in sorted_models]
    weights = np.array(cv_scores) / sum(cv_scores)
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL")
    print("="*60)
    print("\nEnsemble composition:")
    
    ensemble_pred = np.zeros(len(y_test))
    for i, model_result in enumerate(sorted_models):
        pred = model_result['model'].predict(model_result['X_test'])
        ensemble_pred += pred * weights[i]
        print(f"- Alpha {model_result['alpha']}: weight = {weights[i]:.3f}")
    
    # Calculate ensemble metrics
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\nEnsemble Performance:")
    print(f"- R² Score: {ensemble_r2:.4f}")
    print(f"- MAE: ${ensemble_mae:.2f}")
    
    return ensemble_pred, ensemble_r2, ensemble_mae

def plot_predictions_comparison(y_test, best_pred, ensemble_pred, dates, symbol):
    """
    Compare best single model vs ensemble
    """
    plt.figure(figsize=(15, 8))
    
    # Convert to numpy arrays
    y_test_values = y_test.values.flatten()
    
    # Plot 1: Full time series
    plt.subplot(2, 1, 1)
    plt.plot(dates, y_test_values, label='Actual', color='blue', alpha=0.8, linewidth=2)
    plt.plot(dates, best_pred, label='Best Lasso', color='red', alpha=0.6, linewidth=1.5)
    plt.plot(dates, ensemble_pred, label='Ensemble', color='green', alpha=0.6, linewidth=1.5)
    plt.title(f'{symbol} Stock Price Predictions - Full Period')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Last 60 days zoomed
    plt.subplot(2, 1, 2)
    last_60 = -60
    plt.plot(dates[last_60:], y_test_values[last_60:], label='Actual', color='blue', alpha=0.8, linewidth=2, marker='o', markersize=4)
    plt.plot(dates[last_60:], best_pred[last_60:], label='Best Lasso', color='red', alpha=0.6, linewidth=1.5, marker='s', markersize=3)
    plt.plot(dates[last_60:], ensemble_pred[last_60:], label='Ensemble', color='green', alpha=0.6, linewidth=1.5, marker='^', markersize=3)
    plt.title('Last 60 Days - Detailed View')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    filename = f'output_plots/{symbol}_predictions_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved predictions comparison plot: {filename}")
    plt.close()

def main():
    # Parameters
    SYMBOL = 'SPY'  # Change this to any stock symbol
    PREDICTION_DAYS = 5  # Predict 5 days ahead
    
    print(f"\nOptimized Lasso Stock Price Prediction for {SYMBOL}")
    print("=" * 60)
    
    # Step 1: Get data
    print("\n1. Downloading stock and market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years for more data
    data = get_stock_data(SYMBOL, start_date, end_date)
    print(f"Downloaded {len(data)} days of data")
    
    # Step 2: Create features
    print("\n2. Creating optimized features for Lasso...")
    data_with_features = create_optimized_features(data)
    print(f"Created {len(data_with_features.columns)} total features")
    
    # Step 3: Prepare data
    print("\n3. Preparing data for machine learning...")
    
    # Create target variable
    data_with_features['Target'] = data_with_features['Close'].shift(-PREDICTION_DAYS)
    
    # Remove rows with NaN values
    data_with_features = data_with_features.dropna()
    
    # Select features
    exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'SPY', 'VIX', 'DXY']
    feature_columns = [col for col in data_with_features.columns if col not in exclude_cols]
    
    # Remove features with very low variance
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.0001)
    X = data_with_features[feature_columns]
    X_selected = selector.fit_transform(X)
    selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
    
    print(f"Selected {len(selected_features)} features after variance threshold")
    
    # Split data
    X = data_with_features[selected_features]
    y = data_with_features['Target']
    dates = data_with_features.index
    
    # Time series split (80/20)
    split_index = int(len(data_with_features) * 0.8)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    dates_test = dates[split_index:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 4: Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Optimize Lasso alpha
    print("\n5. Optimizing Lasso model...")
    best_result, all_results = optimize_lasso_alpha(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Store scaled data for ensemble
    for result in all_results:
        result['X_test'] = X_test_scaled
    
    # Step 6: Analyze best model features
    best_model = best_result['model']
    feature_analysis = analyze_lasso_features(best_model, selected_features)
    
    # Step 7: Create ensemble predictions
    ensemble_pred, ensemble_r2, ensemble_mae = create_ensemble_predictions(all_results, y_test)
    
    # Step 8: Analyze prediction errors
    best_predictions = best_model.predict(X_test_scaled)
    errors, abs_errors = analyze_prediction_errors(y_test, best_predictions, dates_test, SYMBOL)
    
    # Step 9: Visualize predictions
    print("\n6. Creating visualizations...")
    plot_predictions_comparison(y_test, best_predictions, ensemble_pred, dates_test, SYMBOL)
    
    # Step 10: Make future prediction
    print("\n" + "="*60)
    print("FUTURE PREDICTIONS")
    print("="*60)
    
    latest_features = X_test.iloc[-1:].values
    latest_features_scaled = scaler.transform(latest_features)
    
    # Best single model prediction
    future_prediction = best_model.predict(latest_features_scaled)[0]
    
    # Ensemble prediction
    ensemble_future = np.mean([r['model'].predict(latest_features_scaled)[0] 
                               for r in sorted(all_results, key=lambda x: x['cv_r2_mean'], reverse=True)[:3]])
    
    current_price = data['Close'].iloc[-1]
    
    print(f"\nCurrent {SYMBOL} price: ${current_price:.2f}")
    print(f"Best Lasso prediction in {PREDICTION_DAYS} days: ${future_prediction:.2f}")
    print(f"Ensemble prediction in {PREDICTION_DAYS} days: ${ensemble_future:.2f}")
    print(f"Expected change (Lasso): {((future_prediction - current_price) / current_price * 100):.2f}%")
    print(f"Expected change (Ensemble): {((ensemble_future - current_price) / current_price * 100):.2f}%")
    
    # Confidence interval based on historical errors
    error_std = np.std(errors)
    print(f"\nPrediction confidence interval (±1 std):")
    print(f"Lasso: ${future_prediction - error_std:.2f} - ${future_prediction + error_std:.2f}")
    print(f"Ensemble: ${ensemble_future - error_std:.2f} - ${ensemble_future + error_std:.2f}")
    
    # Save summary report
    with open(f'output_plots/{SYMBOL}_prediction_summary.txt', 'w') as f:
        f.write(f"Optimized Lasso Stock Prediction Summary for {SYMBOL}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Prediction Horizon: {PREDICTION_DAYS} days\n")
        f.write(f"Total Features Created: {len(feature_columns)}\n")
        f.write(f"Features Selected: {len(selected_features)}\n\n")
        
        f.write("Model Performance:\n")
        f.write(f"Best Alpha: {best_result['alpha']}\n")
        f.write(f"CV R² Score: {best_result['cv_r2_mean']:.4f} (+/- {best_result['cv_r2_std']:.4f})\n")
        f.write(f"Test R² Score: {best_result['test_r2']:.4f}\n")
        f.write(f"Test MAE: ${best_result['test_mae']:.2f}\n")
        f.write(f"Features Used: {best_result['non_zero_features']}\n\n")
        
        f.write("Ensemble Performance:\n")
        f.write(f"R² Score: {ensemble_r2:.4f}\n")
        f.write(f"MAE: ${ensemble_mae:.2f}\n\n")
        
        f.write("Top 10 Important Features:\n")
        top_features = feature_analysis[feature_analysis['coefficient'] != 0].head(10)
        for idx, row in top_features.iterrows():
            f.write(f"- {row['feature']}: {row['coefficient']:.4f}\n")
    
    print("\n✅ Analysis complete! Check 'output_plots' folder for detailed visualizations and summary.")

if __name__ == "__main__":
    main()