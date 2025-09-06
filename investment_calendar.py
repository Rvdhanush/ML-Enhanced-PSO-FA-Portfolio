import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import calendar
from accurate_ml_predictor import AccurateMLPredictor
from twelve_data_fetcher import TwelveDataFetcher
import warnings
warnings.filterwarnings('ignore')

class InvestmentCalendar:
    """
    Investment Calendar for weekly/monthly investment planning
    Provides recommendations for when to invest in specific stocks
    """
    
    def __init__(self):
        self.ml_predictor = AccurateMLPredictor()
        self.data_fetcher = TwelveDataFetcher()
        self.recommendations = {}
        
        # Market patterns and seasonality
        self.seasonal_patterns = {
            'TECH': {
                'strong_months': [1, 2, 10, 11, 12],  # January, February, October, November, December
                'weak_months': [5, 6, 7, 8, 9],       # May-September
                'earnings_seasons': [1, 4, 7, 10]      # Quarterly earnings
            },
            'FINANCIAL': {
                'strong_months': [1, 12],
                'weak_months': [6, 7, 8],
                'earnings_seasons': [1, 4, 7, 10]
            },
            'CONSUMER': {
                'strong_months': [11, 12, 1],  # Holiday season
                'weak_months': [2, 3],
                'earnings_seasons': [1, 4, 7, 10]
            }
        }
    
    def generate_weekly_calendar(self, symbols, weeks_ahead=4):
        """Generate weekly investment recommendations"""
        weekly_recommendations = {}
        
        # Get current date
        today = date.today()
        
        for week in range(weeks_ahead):
            week_start = today + timedelta(weeks=week)
            week_end = week_start + timedelta(days=6)
            
            week_key = f"Week_{week+1}_{week_start.strftime('%Y-%m-%d')}"
            weekly_recommendations[week_key] = {
                'date_range': f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}",
                'week_start': week_start,
                'week_end': week_end,
                'recommendations': {}
            }
            
            # Analyze each symbol for this week
            for symbol in symbols:
                try:
                    # Get historical data
                    data = self.data_fetcher.fetch_stock_data(symbol, period='1y')
                    
                    if data is not None and not data.empty:
                        # Train ML model
                        self.ml_predictor.train_model(symbol, data)
                        
                        # Get weekly recommendation
                        weekly_rec = self._analyze_weekly_opportunity(symbol, data, week_start)
                        weekly_recommendations[week_key]['recommendations'][symbol] = weekly_rec
                        
                except Exception as e:
                    print(f"Error analyzing {symbol} for {week_key}: {str(e)}")
                    continue
        
        return weekly_recommendations
    
    def generate_monthly_calendar(self, symbols, months_ahead=6):
        """Generate monthly investment recommendations"""
        monthly_recommendations = {}
        
        # Get current date
        today = date.today()
        
        for month in range(months_ahead):
            # Calculate target month
            target_date = today + timedelta(days=30 * month)
            month_start = target_date.replace(day=1)
            
            # Get last day of month
            last_day = calendar.monthrange(target_date.year, target_date.month)[1]
            month_end = target_date.replace(day=last_day)
            
            month_key = f"Month_{month+1}_{target_date.strftime('%Y-%m')}"
            monthly_recommendations[month_key] = {
                'month_name': target_date.strftime('%B %Y'),
                'date_range': f"{month_start.strftime('%b %d')} - {month_end.strftime('%b %d, %Y')}",
                'month_start': month_start,
                'month_end': month_end,
                'recommendations': {}
            }
            
            # Analyze each symbol for this month
            for symbol in symbols:
                try:
                    # Get historical data
                    data = self.data_fetcher.fetch_stock_data(symbol, period='2y')
                    
                    if data is not None and not data.empty:
                        # Train ML model
                        self.ml_predictor.train_model(symbol, data)
                        
                        # Get monthly recommendation
                        monthly_rec = self._analyze_monthly_opportunity(symbol, data, target_date)
                        monthly_recommendations[month_key]['recommendations'][symbol] = monthly_rec
                        
                except Exception as e:
                    print(f"Error analyzing {symbol} for {month_key}: {str(e)}")
                    continue
        
        return monthly_recommendations
    
    def _analyze_weekly_opportunity(self, symbol, data, target_week):
        """Analyze investment opportunity for a specific week"""
        try:
            # Get current price and recent trends
            current_price = data['Close'].iloc[-1]
            
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
            
            # Weekly trend analysis
            weekly_change = data['Close'].pct_change(5).iloc[-1] * 100
            
            # ML prediction
            predicted_price = self.ml_predictor.predict_price(symbol, data, days_ahead=7)
            
            # Calculate opportunity score (0-100)
            opportunity_score = 50  # Base score
            
            # Technical analysis factors
            if current_price > sma_5 > sma_20:
                opportunity_score += 15  # Uptrend
            elif current_price < sma_5 < sma_20:
                opportunity_score -= 15  # Downtrend
            
            # RSI factors
            if 30 <= rsi <= 70:
                opportunity_score += 10  # Good RSI range
            elif rsi < 30:
                opportunity_score += 20  # Oversold - good buying opportunity
            elif rsi > 70:
                opportunity_score -= 20  # Overbought
            
            # Volatility factors
            if volatility < 0.3:
                opportunity_score += 10  # Low volatility - stable
            elif volatility > 0.5:
                opportunity_score -= 10  # High volatility - risky
            
            # ML prediction factor
            if predicted_price and predicted_price > current_price * 1.02:
                opportunity_score += 15  # ML predicts growth
            elif predicted_price and predicted_price < current_price * 0.98:
                opportunity_score -= 15  # ML predicts decline
            
            # Seasonal factors
            opportunity_score += self._get_seasonal_score(symbol, target_week)
            
            # Ensure score is within bounds
            opportunity_score = max(0, min(100, opportunity_score))
            
            # Determine recommendation
            if opportunity_score >= 75:
                recommendation = "🟢 STRONG BUY"
                action = "Excellent buying opportunity"
            elif opportunity_score >= 60:
                recommendation = "🟢 BUY"
                action = "Good time to invest"
            elif opportunity_score >= 40:
                recommendation = "🟡 HOLD/WAIT"
                action = "Monitor closely, wait for better entry"
            elif opportunity_score >= 25:
                recommendation = "🔴 AVOID"
                action = "Not recommended this week"
            else:
                recommendation = "🔴 STRONG AVOID"
                action = "High risk, avoid investment"
            
            return {
                'recommendation': recommendation,
                'action': action,
                'opportunity_score': opportunity_score,
                'current_price': f"${current_price:.2f}",
                'predicted_price': f"${predicted_price:.2f}" if predicted_price else "N/A",
                'weekly_change': f"{weekly_change:.1f}%",
                'rsi': f"{rsi:.1f}",
                'volatility': f"{volatility:.1%}",
                'key_factors': self._get_key_factors(opportunity_score, rsi, volatility, weekly_change)
            }
            
        except Exception as e:
            return {
                'recommendation': "❌ ERROR",
                'action': f"Analysis failed: {str(e)}",
                'opportunity_score': 0,
                'current_price': "N/A",
                'predicted_price': "N/A",
                'weekly_change': "N/A",
                'rsi': "N/A",
                'volatility': "N/A",
                'key_factors': []
            }
    
    def _analyze_monthly_opportunity(self, symbol, data, target_month):
        """Analyze investment opportunity for a specific month"""
        try:
            # Similar to weekly analysis but with longer timeframes
            current_price = data['Close'].iloc[-1]
            
            # Monthly technical indicators
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            # Monthly trend
            monthly_change = data['Close'].pct_change(21).iloc[-1] * 100  # ~1 month
            
            # Long-term volatility
            volatility = data['Close'].pct_change().rolling(60).std().iloc[-1] * np.sqrt(252)
            
            # ML prediction for 30 days
            predicted_price = self.ml_predictor.predict_price(symbol, data, days_ahead=30)
            
            # Monthly opportunity score
            opportunity_score = 50
            
            # Long-term trend analysis
            if current_price > sma_20 > sma_50:
                opportunity_score += 20
            elif current_price < sma_20 < sma_50:
                opportunity_score -= 20
            
            # Monthly momentum
            if monthly_change > 5:
                opportunity_score += 15
            elif monthly_change < -5:
                opportunity_score -= 15
            
            # ML prediction (stronger weight for monthly)
            if predicted_price and predicted_price > current_price * 1.05:
                opportunity_score += 25
            elif predicted_price and predicted_price < current_price * 0.95:
                opportunity_score -= 25
            
            # Strong seasonal factors for monthly
            seasonal_score = self._get_seasonal_score(symbol, target_month) * 2
            opportunity_score += seasonal_score
            
            # Ensure bounds
            opportunity_score = max(0, min(100, opportunity_score))
            
            # Monthly recommendations
            if opportunity_score >= 80:
                recommendation = "🚀 EXCELLENT MONTH"
                action = "Prime investment opportunity"
            elif opportunity_score >= 65:
                recommendation = "🟢 GOOD MONTH"
                action = "Favorable conditions for investment"
            elif opportunity_score >= 45:
                recommendation = "🟡 NEUTRAL MONTH"
                action = "Average conditions, proceed with caution"
            elif opportunity_score >= 30:
                recommendation = "🔴 POOR MONTH"
                action = "Unfavorable conditions"
            else:
                recommendation = "🚫 AVOID THIS MONTH"
                action = "High risk period, avoid investment"
            
            return {
                'recommendation': recommendation,
                'action': action,
                'opportunity_score': opportunity_score,
                'current_price': f"${current_price:.2f}",
                'predicted_price': f"${predicted_price:.2f}" if predicted_price else "N/A",
                'monthly_change': f"{monthly_change:.1f}%",
                'volatility': f"{volatility:.1%}",
                'seasonal_factors': self._get_seasonal_description(symbol, target_month),
                'key_factors': self._get_monthly_factors(opportunity_score, monthly_change, volatility)
            }
            
        except Exception as e:
            return {
                'recommendation': "❌ ERROR",
                'action': f"Analysis failed: {str(e)}",
                'opportunity_score': 0,
                'current_price': "N/A",
                'predicted_price': "N/A",
                'monthly_change': "N/A",
                'volatility': "N/A",
                'seasonal_factors': "N/A",
                'key_factors': []
            }
    
    def _get_seasonal_score(self, symbol, target_date):
        """Get seasonal score based on historical patterns"""
        month = target_date.month if hasattr(target_date, 'month') else target_date.month
        
        # Determine sector (simplified)
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'TSLA']
        financial_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        
        if symbol in tech_symbols:
            sector = 'TECH'
        elif symbol in financial_symbols:
            sector = 'FINANCIAL'
        else:
            sector = 'CONSUMER'
        
        patterns = self.seasonal_patterns.get(sector, self.seasonal_patterns['CONSUMER'])
        
        if month in patterns['strong_months']:
            return 10
        elif month in patterns['weak_months']:
            return -10
        elif month in patterns['earnings_seasons']:
            return 5
        else:
            return 0
    
    def _get_seasonal_description(self, symbol, target_date):
        """Get seasonal description for the month"""
        month = target_date.month if hasattr(target_date, 'month') else target_date.month
        month_name = calendar.month_name[month]
        
        # Seasonal patterns description
        if month in [11, 12, 1]:
            return f"{month_name}: Holiday season - typically strong for consumer stocks"
        elif month in [4, 10]:
            return f"{month_name}: Earnings season - increased volatility expected"
        elif month in [5, 6, 7, 8]:
            return f"{month_name}: Summer months - traditionally weaker for tech stocks"
        else:
            return f"{month_name}: Normal trading conditions"
    
    def _get_key_factors(self, score, rsi, volatility, weekly_change):
        """Get key factors affecting the recommendation"""
        factors = []
        
        if score >= 75:
            factors.append("Strong technical signals")
        elif score <= 25:
            factors.append("Weak technical signals")
        
        if rsi < 30:
            factors.append("Oversold condition - potential bounce")
        elif rsi > 70:
            factors.append("Overbought condition - potential pullback")
        
        if volatility > 0.5:
            factors.append("High volatility - increased risk")
        elif volatility < 0.2:
            factors.append("Low volatility - stable conditions")
        
        if abs(weekly_change) > 5:
            factors.append("Strong recent momentum")
        
        return factors
    
    def _get_monthly_factors(self, score, monthly_change, volatility):
        """Get key factors for monthly analysis"""
        factors = []
        
        if score >= 80:
            factors.append("Excellent long-term outlook")
        elif score <= 20:
            factors.append("Poor long-term outlook")
        
        if abs(monthly_change) > 10:
            factors.append("Strong monthly momentum")
        
        if volatility > 0.4:
            factors.append("High volatility - consider dollar-cost averaging")
        
        return factors
    
    def get_best_investment_opportunities(self, symbols, timeframe='weekly'):
        """Get the best investment opportunities across all symbols"""
        if timeframe == 'weekly':
            calendar_data = self.generate_weekly_calendar(symbols, weeks_ahead=4)
        else:
            calendar_data = self.generate_monthly_calendar(symbols, months_ahead=6)
        
        opportunities = []
        
        for period_key, period_data in calendar_data.items():
            for symbol, recommendation in period_data['recommendations'].items():
                if recommendation['opportunity_score'] >= 60:  # Only good opportunities
                    opportunities.append({
                        'symbol': symbol,
                        'period': period_data['date_range'],
                        'recommendation': recommendation['recommendation'],
                        'score': recommendation['opportunity_score'],
                        'action': recommendation['action']
                    })
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities
