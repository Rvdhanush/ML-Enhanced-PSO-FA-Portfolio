import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from accurate_ml_predictor import AccurateMLPredictor
import warnings
import time
import random
warnings.filterwarnings('ignore')

class HybridPSOFAOptimizer:
    """Hybrid PSO-FA optimizer enhanced with ML predictions"""
    
    def __init__(self, n_particles=50, max_iter=100):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.best_solution = None
        self.best_fitness = float('inf')
        self.optimization_history = []
        
    def optimize(self, returns, risk_free_rate=0.02, ml_predictions=None):
        """Optimize portfolio weights using PSO-FA with ML enhancements"""
        n_assets = returns.shape[1]
        
        # Initialize particles (portfolio weights)
        particles = np.random.rand(self.n_particles, n_assets)
        particles = particles / np.sum(particles, axis=1, keepdims=True)  # Normalize
        
        velocities = np.random.randn(self.n_particles, n_assets) * 0.1
        
        # Initialize personal and global best
        personal_best = particles.copy()
        personal_best_values = np.array([self._objective_function(p, returns, risk_free_rate, ml_predictions) 
                                       for p in particles])
        global_best_idx = np.argmin(personal_best_values)
        global_best = particles[global_best_idx].copy()
        self.best_fitness = personal_best_values[global_best_idx]
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 2.0  # Cognitive parameter
        c2 = 2.0  # Social parameter
        
        # FA parameters
        alpha = 0.5  # Randomization parameter
        beta = 1.0   # Attraction parameter
        gamma = 1.0  # Absorption parameter
        
        for iteration in range(self.max_iter):
            # PSO phase
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Normalize weights and ensure constraints
                particles[i] = np.maximum(particles[i], 0)  # No short selling
                particles[i] = particles[i] / np.sum(particles[i])
            
            # FA phase (every 10 iterations)
            if iteration % 10 == 0:
                for i in range(self.n_particles):
                    for j in range(self.n_particles):
                        if i != j:
                            # Calculate distance
                            distance = np.linalg.norm(particles[i] - particles[j])
                            
                            # Firefly attraction
                            if personal_best_values[j] < personal_best_values[i]:
                                # Move towards better solution
                                particles[i] += beta * np.exp(-gamma * distance**2) * (particles[j] - particles[i])
                                
                                # Add randomization
                                particles[i] += alpha * np.random.randn(n_assets) * 0.1
                                
                                # Normalize weights
                                particles[i] = np.maximum(particles[i], 0)
                                particles[i] = particles[i] / np.sum(particles[i])
            
            # Update personal and global best
            for i in range(self.n_particles):
                current_value = self._objective_function(particles[i], returns, risk_free_rate, ml_predictions)
                if current_value < personal_best_values[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_values[i] = current_value
                    
                    if current_value < self.best_fitness:
                        global_best = particles[i].copy()
                        global_best_idx = i
                        self.best_fitness = current_value
            
            # Store optimization history
            self.optimization_history.append({
                'iteration': iteration,
                'best_fitness': self.best_fitness,
                'global_best': global_best.copy()
            })
            
            # Adaptive parameters
            w = 0.9 - 0.5 * iteration / self.max_iter  # Decreasing inertia
            alpha = 0.5 * (1 - iteration / self.max_iter)  # Decreasing randomization
        
        self.best_solution = global_best
        return global_best
    
    def _objective_function(self, weights, returns, risk_free_rate, ml_predictions=None):
        """Objective function: minimize negative Sharpe ratio with ML enhancements"""
        # Calculate portfolio returns
        portfolio_returns = np.sum(returns * weights, axis=1)
        
        # Calculate expected return and volatility
        expected_return = np.mean(portfolio_returns) * 252  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        
        # Base Sharpe ratio
        sharpe_ratio = (expected_return - risk_free_rate) / (volatility + 1e-8)
        
        # ML enhancement if available
        if ml_predictions is not None:
            # Get ML confidence scores
            ml_confidence = np.mean([pred.get('confidence_score', 0.5) for pred in ml_predictions.values()])
            
            # Adjust Sharpe ratio based on ML confidence
            ml_adjusted_sharpe = sharpe_ratio * (0.5 + 0.5 * ml_confidence)
            
            # Add ML prediction accuracy bonus
            prediction_bonus = 0
            for pred in ml_predictions.values():
                if pred.get('price_trend') == 'Bullish':
                    prediction_bonus += 0.1
                elif pred.get('price_trend') == 'Bearish':
                    prediction_bonus -= 0.1
            
            final_sharpe = ml_adjusted_sharpe + prediction_bonus
        else:
            final_sharpe = sharpe_ratio
        
        return -final_sharpe  # Minimize negative Sharpe ratio

class PortfolioOptimizer:
    """Main portfolio optimizer that integrates ML/DL with traditional optimization"""
    
    def __init__(self, stocks, start_date, end_date):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.risk_free_rate = 0.02
        
        # ML-enhanced optimizer
        self.ml_optimizer = AccurateMLPredictor()
        self.trained_models = None
        self.ml_predictions = None
        
    def fetch_data(self):
        """Fetch historical data for the selected stocks with enhanced error handling and rate limiting"""
        try:
            print(f"🔄 Fetching data for {len(self.stocks)} stocks...")
            
            # Try to fetch data for each stock
            successful_stocks = []
            stock_data = {}
            
            for i, stock in enumerate(self.stocks):
                try:
                    print(f"📊 Fetching {stock}... ({i+1}/{len(self.stocks)})")
                    ticker = yf.Ticker(stock)
                    
                    # Add delay between requests to avoid rate limiting
                    if i > 0:
                        delay = random.uniform(1.0, 3.0)  # Random delay 1-3 seconds
                        print(f"⏳ Waiting {delay:.1f}s to avoid rate limiting...")
                        time.sleep(delay)
                    
                    # Try multiple data fetching strategies
                    data = None
                    
                    # Strategy 1: Try specific date range with shorter period
                    try:
                        # Use a more reasonable date range
                        start_date = pd.to_datetime(self.start_date)
                        end_date = pd.to_datetime(self.end_date)
                        
                        # If date range is too large, limit it
                        if (end_date - start_date).days > 365:
                            start_date = end_date - pd.Timedelta(days=365)
                            print(f"📅 Adjusted date range to last 365 days for {stock}")
                        
                        data = ticker.history(start=start_date, end=end_date, interval='1d')
                        if not data.empty and len(data) > 50:  # Need sufficient data
                            print(f"✅ {stock}: Got {len(data)} data points (date range)")
                        else:
                            data = None
                    except Exception as e:
                        print(f"⚠️ {stock}: Date range strategy failed - {str(e)}")
                        data = None
                    
                    # Strategy 2: Try period-based fetching if date range fails
                    if data is None or data.empty:
                        try:
                            print(f"🔄 Trying period-based strategy for {stock}...")
                            data = ticker.history(period="1y", interval='1d')
                            if not data.empty and len(data) > 50:
                                print(f"✅ {stock}: Got {len(data)} data points (1y period)")
                            else:
                                data = None
                        except Exception as e:
                            print(f"⚠️ {stock}: Period strategy failed - {str(e)}")
                            data = None
                    
                    # Strategy 3: Try 6-month period if 1y fails
                    if data is None or data.empty:
                        try:
                            print(f"🔄 Trying 6-month strategy for {stock}...")
                            data = ticker.history(period="6mo", interval='1d')
                            if not data.empty and len(data) > 50:
                                print(f"✅ {stock}: Got {len(data)} data points (6mo period)")
                            else:
                                data = None
                        except Exception as e:
                            print(f"⚠️ {stock}: 6-month strategy failed - {str(e)}")
                            data = None
                    
                    if data is not None and not data.empty and len(data) > 50:
                        stock_data[stock] = data
                        successful_stocks.append(stock)
                        print(f"🎯 {stock}: Successfully added to portfolio")
                    else:
                        print(f"❌ {stock}: Insufficient data ({len(data) if data is not None else 0} points)")
                        
                except Exception as e:
                    print(f"❌ {stock}: Error fetching data - {str(e)}")
                    continue
            
            if not successful_stocks:
                raise ValueError("No data available for any of the selected stocks. Please try different stocks or check your internet connection.")
            
            print(f"🎯 Successfully fetched data for {len(successful_stocks)} stocks: {', '.join(successful_stocks)}")
            
            # Update stocks list to only include successful ones
            self.stocks = successful_stocks
            
            # Calculate returns for successful stocks
            self.returns = pd.DataFrame()
            for stock in self.stocks:
                if stock in stock_data:
                    # Calculate daily returns
                    stock_returns = stock_data[stock]['Close'].pct_change().dropna()
                    self.returns[stock] = stock_returns
            
            # Align all returns to same date range
            self.returns = self.returns.dropna()
            
            if self.returns.empty:
                raise ValueError("No overlapping data found for the selected stocks")
            
            # Calculate mean returns and covariance matrix
            self.mean_returns = self.returns.mean()
            self.cov_matrix = self.returns.cov()
            
            print(f"📈 Data ready: {len(self.returns)} trading days, {len(self.stocks)} stocks")
            return self.returns
            
        except Exception as e:
            print(f"❌ Error in fetch_data: {str(e)}")
            # Return empty DataFrame to trigger error handling in the UI
            return pd.DataFrame()
    
    def optimize_portfolio(self, method='hybrid', use_ml=True):
        """Optimize portfolio using specified method with ML enhancement"""
        if self.returns is None:
            self.fetch_data()
        
        if method == 'hybrid' and use_ml:
            # ML-enhanced optimization
            return self._optimize_with_ml()
        elif method == 'hybrid':
            # Traditional PSO-FA without ML
            return self._optimize_traditional()
        else:
            # Traditional mean-variance optimization
            return self._optimize_traditional_mvo()
    
    def _optimize_with_ml(self):
        """Optimize portfolio using ML predictions integrated with PSO-FA"""
        print("Training ML models and optimizing portfolio...")
        
        # Prepare stock data for ML models
        stock_data = {}
        for stock in self.stocks:
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(start=self.start_date, end=self.end_date)
                if not hist.empty:
                    stock_data[stock] = hist
            except Exception as e:
                print(f"Error preparing ML data for {stock}: {str(e)}")
                continue
        
        if not stock_data:
            raise ValueError("No data available for ML optimization")
        
        # Train ML models and get predictions
        ml_predictions = {}
        trained_models = {}
        
        for symbol, data in stock_data.items():
            try:
                # Prepare features and train model
                X, y = self.ml_optimizer.prepare_features(data)
                if X is not None and y is not None:
                    model = self.ml_optimizer.train_ensemble_model(X, y)
                    prediction = self.ml_optimizer.predict_future_prices(data, model)
                    
                    ml_predictions[symbol] = prediction
                    trained_models[symbol] = model
            except Exception as e:
                print(f"Error training ML model for {symbol}: {str(e)}")
                continue
        
        # Use traditional optimization if no ML predictions
        if not ml_predictions:
            return self._optimize_traditional()
        
        # Optimize with ML predictions
        optimizer = HybridPSOFAOptimizer(n_particles=50, max_iter=100)
        weights = optimizer.optimize(self.returns, self.risk_free_rate, ml_predictions)
        
        # Store ML results
        self.trained_models = trained_models
        self.ml_predictions = ml_predictions
        
        return weights
    
    def _optimize_traditional(self):
        """Traditional PSO-FA optimization without ML"""
        optimizer = HybridPSOFAOptimizer(n_particles=50, max_iter=100)
        weights = optimizer.optimize(self.returns, self.risk_free_rate)
        return weights
    
    def _optimize_traditional_mvo(self):
        """Traditional mean-variance optimization"""
        n_assets = len(self.stocks)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            return portfolio_variance
        
        # Constraints: weights sum to 1, no short selling
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            raise ValueError("Traditional optimization failed")
    
    def get_portfolio_metrics(self, weights):
        """Calculate portfolio performance metrics"""
        if self.returns is None:
            self.fetch_data()
        
        # Calculate portfolio returns
        portfolio_returns = np.sum(self.returns * weights, axis=1)
        
        # Basic metrics
        annual_return = np.mean(portfolio_returns) * 252
        annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Information ratio (if benchmark available)
        # For simplicity, using risk-free rate as benchmark
        excess_returns = portfolio_returns - self.risk_free_rate/252
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # ML-enhanced metrics if available
        ml_metrics = {}
        if self.ml_predictions is not None:
            ml_metrics = self._calculate_ml_metrics(weights)
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'ml_metrics': ml_metrics
        }
    
    def _calculate_ml_metrics(self, weights):
        """Calculate ML-specific portfolio metrics"""
        if not self.ml_predictions:
            return {}
        
        # ML confidence score
        confidence_scores = []
        for pred in self.ml_predictions.values():
            if 'confidence_score' in pred:
                confidence_scores.append(pred['confidence_score'])
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # ML prediction accuracy (simplified)
        bullish_count = sum(1 for pred in self.ml_predictions.values() 
                           if pred.get('price_trend') == 'Bullish')
        prediction_ratio = bullish_count / len(self.ml_predictions) if self.ml_predictions else 0
        
        # ML risk assessment
        high_risk_count = sum(1 for pred in self.ml_predictions.values() 
                             if pred.get('risk_level') == 'High')
        risk_ratio = high_risk_count / len(self.ml_predictions) if self.ml_predictions else 0
        
        return {
            'ml_confidence': avg_confidence,
            'prediction_ratio': prediction_ratio,
            'risk_ratio': risk_ratio,
            'ml_enhanced_sharpe': self._calculate_ml_enhanced_sharpe(weights)
        }
    
    def _calculate_ml_enhanced_sharpe(self, weights):
        """Calculate Sharpe ratio enhanced with ML predictions"""
        if not self.ml_predictions:
            return 0
        
        # Base portfolio metrics
        portfolio_returns = np.sum(self.returns * weights, axis=1)
        annual_return = np.mean(portfolio_returns) * 252
        annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # ML adjustment factor
        ml_adjustment = 0
        for pred in self.ml_predictions.values():
            if pred.get('price_trend') == 'Bullish':
                ml_adjustment += 0.1
            elif pred.get('price_trend') == 'Bearish':
                ml_adjustment -= 0.1
        
        # Adjusted return
        adjusted_return = annual_return + ml_adjustment
        
        # ML-enhanced Sharpe ratio
        ml_sharpe = (adjusted_return - self.risk_free_rate) / annual_volatility
        
        return ml_sharpe
    
    def get_risk_decomposition(self, weights):
        """Calculate risk contribution by asset"""
        if self.returns is None:
            self.fetch_data()
        
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        marginal_risk = np.dot(self.cov_matrix, weights) / np.sqrt(portfolio_variance)
        risk_contribution = weights * marginal_risk
        
        return dict(zip(self.stocks, risk_contribution))
    
    def get_correlation_matrix(self):
        """Get correlation matrix of returns"""
        if self.returns is None:
            self.fetch_data()
        
        return self.returns.corr()
    
    def run_monte_carlo(self, weights, n_simulations=1000, time_horizon=1):
        """Run Monte Carlo simulation with ML enhancements"""
        if self.returns is None:
            self.fetch_data()
        
        # Calculate daily returns and volatility
        portfolio_returns = np.sum(self.returns * weights, axis=1)
        daily_return = np.mean(portfolio_returns)
        daily_vol = np.std(portfolio_returns)
        
        # ML enhancement for volatility prediction
        if self.ml_predictions:
            ml_volatility_adjustment = np.mean([pred.get('volatility_forecast', 1.0) 
                                              for pred in self.ml_predictions.values()])
            daily_vol *= ml_volatility_adjustment
        
        # Run simulation
        days = int(time_horizon * 252)
        simulations = np.zeros((n_simulations, days))
        
        for i in range(n_simulations):
            # Generate random returns with ML-adjusted volatility
            sim_returns = np.random.normal(daily_return, daily_vol, days)
            simulations[i] = np.cumprod(1 + sim_returns) - 1
        
        return simulations
    
    def get_ml_insights(self):
        """Get comprehensive ML insights for the portfolio"""
        if not self.ml_predictions:
            return {}
        
        insights = {}
        for symbol, pred in self.ml_predictions.items():
            insights[symbol] = {
                'price_trend': pred.get('price_trend', 'Unknown'),
                'predicted_return': pred.get('ensemble_prediction', 0),
                'risk_level': pred.get('risk_level', 'Unknown'),
                'confidence_score': pred.get('ml_confidence', 0),
                'model_performance': self._get_model_performance(symbol)
            }
        
        return insights
    
    def _get_model_performance(self, symbol):
        """Get performance metrics for ML models"""
        if not self.trained_models or symbol not in self.trained_models:
            return {}
        
        return self.trained_models[symbol].get('performance', {}) 
    
    def _fetch_with_retry(self, ticker, strategy_func, max_retries=3):
        """Fetch data with retry mechanism and exponential backoff"""
        for attempt in range(max_retries):
            try:
                return strategy_func()
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(1, 3)  # Exponential backoff
                        print(f"⏳ Rate limited, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Max retries reached for rate limiting")
                        return None
                else:
                    # Non-rate-limit error, don't retry
                    return None
        return None 