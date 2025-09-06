import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AccurateMLPredictor:
    """
    Accurate ML predictor using REAL stock data only
    No fake data, no random generation - only real market data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def prepare_features(self, data):
        """Prepare technical indicators from REAL stock data"""
        if data is None or data.empty:
            return None, None
            
        df = data.copy()
        
        # Technical indicators using REAL data
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Price momentum
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # High-Low spread
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Remove NaN values
        df = df.dropna()
        
        if df.empty:
            return None, None
        
        # Feature columns
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
            'Volume_Ratio', 'Volatility', 'HL_Spread'
        ]
        
        # Prepare features and target
        X = df[feature_columns].values
        y = df['Close'].shift(-1).dropna().values  # Predict next day's close
        
        # Align X and y
        X = X[:-1]  # Remove last row to match y
        
        return X, y
    
    def train_model(self, symbol, data):
        """Train ML model using REAL stock data"""
        print(f"🤖 Training accurate ML model for {symbol} using REAL data...")
        
        X, y = self.prepare_features(data)
        
        if X is None or len(X) < 50:
            print(f"❌ {symbol}: Insufficient REAL data for training")
            return False
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        
        # Store models and metrics
        self.models[symbol] = {
            'rf': rf_model,
            'gb': gb_model
        }
        self.scalers[symbol] = scaler
        self.performance_metrics[symbol] = {
            'mse': mse,
            'r2': r2,
            'rmse': rmse,
            'data_points': len(X)
        }
        
        print(f"✅ {symbol}: Model trained - R²: {r2:.3f}, RMSE: {rmse:.2f}")
        return True
    
    def predict_price(self, symbol, data, days_ahead=1):
        """Predict future price using trained model"""
        if symbol not in self.models:
            return None
        
        X, _ = self.prepare_features(data)
        
        if X is None or len(X) == 0:
            return None
        
        # Get latest features
        latest_features = X[-1:].reshape(1, -1)
        
        # Scale features
        scaler = self.scalers[symbol]
        latest_features_scaled = scaler.transform(latest_features)
        
        # Make prediction using ensemble
        models = self.models[symbol]
        rf_pred = models['rf'].predict(latest_features_scaled)[0]
        gb_pred = models['gb'].predict(latest_features_scaled)[0]
        
        # Ensemble prediction
        predicted_price = (rf_pred + gb_pred) / 2
        
        return predicted_price
    
    def get_investment_recommendation(self, symbol, data):
        """Get investment recommendation based on REAL data analysis"""
        if data is None or data.empty or len(data) < 20:
            return "❌ Insufficient REAL data for recommendation"
        
        # Get current price and recent trend
        current_price = data['Close'].iloc[-1]
        recent_prices = data['Close'].tail(20)
        
        # Calculate technical indicators
        sma_5 = data['Close'].rolling(5).mean().iloc[-1]
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50
        
        # Volatility
        volatility = data['Close'].pct_change().std() * np.sqrt(252)
        
        # Price trend
        price_trend = (current_price - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        # Volume trend
        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # ML prediction if model is trained
        predicted_price = None
        if symbol in self.models:
            predicted_price = self.predict_price(symbol, data)
        
        # Make recommendation based on REAL data
        signals = []
        
        # Trend signals
        if current_price > sma_5 > sma_20:
            signals.append("bullish_trend")
        elif current_price < sma_5 < sma_20:
            signals.append("bearish_trend")
        
        # RSI signals
        if rsi < 30:
            signals.append("oversold")
        elif rsi > 70:
            signals.append("overbought")
        
        # Volume signals
        if volume_ratio > 1.5:
            signals.append("high_volume")
        
        # ML prediction signal
        if predicted_price and predicted_price > current_price * 1.02:
            signals.append("ml_bullish")
        elif predicted_price and predicted_price < current_price * 0.98:
            signals.append("ml_bearish")
        
        # Generate recommendation
        bullish_signals = sum(1 for s in signals if s in ['bullish_trend', 'oversold', 'ml_bullish'])
        bearish_signals = sum(1 for s in signals if s in ['bearish_trend', 'overbought', 'ml_bearish'])
        
        confidence = "High" if symbol in self.models else "Medium"
        
        if bullish_signals >= 2 and volatility < 0.4:
            recommendation = "🟢 BUY"
            reason = f"Strong bullish signals, low volatility ({volatility:.1%})"
        elif bearish_signals >= 2:
            recommendation = "🔴 SELL"
            reason = f"Strong bearish signals, RSI: {rsi:.1f}"
        elif volatility > 0.5:
            recommendation = "⚠️ HOLD"
            reason = f"High volatility ({volatility:.1%}), wait for stability"
        else:
            recommendation = "🟡 HOLD"
            reason = "Mixed signals, maintain current position"
        
        return {
            'recommendation': recommendation,
            'reason': reason,
            'confidence': confidence,
            'current_price': f"${current_price:.2f}",
            'predicted_price': f"${predicted_price:.2f}" if predicted_price else "N/A",
            'rsi': f"{rsi:.1f}",
            'volatility': f"{volatility:.1%}",
            'trend': f"{price_trend:.1%}"
        }
    
    def get_model_performance(self, symbol):
        """Get model performance metrics"""
        if symbol in self.performance_metrics:
            return self.performance_metrics[symbol]
        return None
