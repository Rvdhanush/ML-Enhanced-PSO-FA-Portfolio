import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

class RealTimeMarketAnalyzer:
    """
    Real-time market analysis with timezone support
    Updates market data every minute for live analysis
    """
    
    def __init__(self):
        self.market_data = {}
        self.analysis_cache = {}
        self.last_update = {}
        self.is_running = False
        
        # Market timezones
        self.market_timezones = {
            'US': 'America/New_York',     # NYSE, NASDAQ
            'EU': 'Europe/London',        # LSE
            'ASIA': 'Asia/Tokyo',         # TSE
            'INDIA': 'Asia/Kolkata',      # NSE, BSE
            'CHINA': 'Asia/Shanghai',     # SSE
            'AUSTRALIA': 'Australia/Sydney' # ASX
        }
        
        # Market hours (in local timezone)
        self.market_hours = {
            'US': {'open': '09:30', 'close': '16:00'},
            'EU': {'open': '08:00', 'close': '16:30'},
            'ASIA': {'open': '09:00', 'close': '15:00'},
            'INDIA': {'open': '09:15', 'close': '15:30'},
            'CHINA': {'open': '09:30', 'close': '15:00'},
            'AUSTRALIA': {'open': '10:00', 'close': '16:00'}
        }
    
    def get_market_status(self, market='US'):
        """Get current market status with timezone awareness"""
        try:
            market_tz = pytz.timezone(self.market_timezones[market])
            current_time = datetime.now(market_tz)
            
            # Check if it's a weekday
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return {
                    'status': 'CLOSED',
                    'reason': 'Weekend',
                    'local_time': current_time.strftime('%H:%M:%S %Z'),
                    'next_open': self._get_next_market_open(market)
                }
            
            # Check market hours
            open_time = datetime.strptime(self.market_hours[market]['open'], '%H:%M').time()
            close_time = datetime.strptime(self.market_hours[market]['close'], '%H:%M').time()
            current_time_only = current_time.time()
            
            if open_time <= current_time_only <= close_time:
                return {
                    'status': 'OPEN',
                    'reason': 'Regular Trading Hours',
                    'local_time': current_time.strftime('%H:%M:%S %Z'),
                    'time_to_close': self._time_until_close(current_time, close_time)
                }
            else:
                return {
                    'status': 'CLOSED',
                    'reason': 'After Hours' if current_time_only > close_time else 'Pre-Market',
                    'local_time': current_time.strftime('%H:%M:%S %Z'),
                    'next_open': self._get_next_market_open(market)
                }
                
        except Exception as e:
            return {
                'status': 'UNKNOWN',
                'reason': f'Error: {str(e)}',
                'local_time': 'N/A',
                'next_open': 'N/A'
            }
    
    def _get_next_market_open(self, market):
        """Calculate next market opening time"""
        try:
            market_tz = pytz.timezone(self.market_timezones[market])
            current_time = datetime.now(market_tz)
            
            # If it's weekend, next open is Monday
            if current_time.weekday() >= 5:
                days_until_monday = 7 - current_time.weekday()
                next_open = current_time + timedelta(days=days_until_monday)
            else:
                # If market is closed today, next open is tomorrow (or today if before open)
                open_time = datetime.strptime(self.market_hours[market]['open'], '%H:%M').time()
                if current_time.time() > datetime.strptime(self.market_hours[market]['close'], '%H:%M').time():
                    next_open = current_time + timedelta(days=1)
                else:
                    next_open = current_time
            
            next_open = next_open.replace(
                hour=int(self.market_hours[market]['open'].split(':')[0]),
                minute=int(self.market_hours[market]['open'].split(':')[1]),
                second=0,
                microsecond=0
            )
            
            return next_open.strftime('%Y-%m-%d %H:%M %Z')
            
        except:
            return 'N/A'
    
    def _time_until_close(self, current_time, close_time):
        """Calculate time until market close"""
        try:
            close_datetime = current_time.replace(
                hour=close_time.hour,
                minute=close_time.minute,
                second=0,
                microsecond=0
            )
            
            time_diff = close_datetime - current_time
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            
            return f"{int(hours)}h {int(minutes)}m"
        except:
            return 'N/A'
    
    def get_realtime_data(self, symbol):
        """Get real-time data for a symbol with caching"""
        try:
            # Check cache first (avoid repeated API calls)
            current_time = datetime.now()
            if symbol in self.last_update:
                time_diff = (current_time - self.last_update[symbol]).total_seconds()
                if time_diff < 60:  # Use cached data if less than 1 minute old
                    cached_data = self.market_data.get(symbol)
                    if cached_data:
                        print(f"📋 Using cached data for {symbol} (age: {time_diff:.0f}s)")
                        return cached_data
            
            # Try multiple real-time sources (reliable system first)
            sources = [
                self._get_yahoo_realtime,  # Now uses our reliable multi-source system
                self._get_finnhub_realtime,
                self._get_alpha_vantage_realtime
            ]
            
            for source in sources:
                try:
                    data = source(symbol)
                    if data:
                        # Cache the data
                        self.market_data[symbol] = data
                        self.last_update[symbol] = current_time
                        return data
                except Exception as e:
                    print(f"Source error for {symbol}: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"Real-time data error for {symbol}: {str(e)}")
            return None
    
    def _get_yahoo_realtime(self, symbol):
        """Get real-time data from our reliable multi-source system"""
        try:
            # Use our reliable data fetcher instead of direct Yahoo Finance
            from twelve_data_fetcher import TwelveDataFetcher
            
            fetcher = TwelveDataFetcher()
            
            # Get recent data (last few days) to extract current price
            data = fetcher.fetch_stock_data(symbol, period='5d')
            
            if data is not None and not data.empty:
                # Get latest data point
                latest = data.iloc[-1]
                previous = data.iloc[-2] if len(data) > 1 else latest
                
                current_price = latest['Close']
                change = current_price - previous['Close']
                change_percent = (change / previous['Close']) * 100 if previous['Close'] != 0 else 0
                
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': latest['Volume'],
                    'day_high': latest['High'],
                    'day_low': latest['Low'],
                    'open': latest['Open'],
                    'timestamp': datetime.now().isoformat(),
                    'source': 'reliable_multi_source'
                }
        except Exception as e:
            print(f"Reliable source error for {symbol}: {str(e)}")
        return None
    
    def _get_finnhub_realtime(self, symbol):
        """Get real-time data from Finnhub"""
        try:
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                'symbol': symbol,
                'token': 'demo'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'c' in data and data['c'] > 0:
                    return {
                        'symbol': symbol,
                        'price': data.get('c', 0),  # Current price
                        'change': data.get('d', 0),  # Change
                        'change_percent': data.get('dp', 0),  # Change percent
                        'day_high': data.get('h', 0),  # High
                        'day_low': data.get('l', 0),   # Low
                        'open': data.get('o', 0),     # Open
                        'previous_close': data.get('pc', 0),  # Previous close
                        'timestamp': datetime.now().isoformat(),
                        'source': 'finnhub'
                    }
        except:
            pass
        return None
    
    def _get_alpha_vantage_realtime(self, symbol):
        """Get real-time data from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': 'demo'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    return {
                        'symbol': symbol,
                        'price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                        'volume': int(quote.get('06. volume', 0)),
                        'day_high': float(quote.get('03. high', 0)),
                        'day_low': float(quote.get('04. low', 0)),
                        'open': float(quote.get('02. open', 0)),
                        'previous_close': float(quote.get('08. previous close', 0)),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'alpha_vantage'
                    }
        except:
            pass
        return None
    
    def analyze_realtime_trends(self, symbols, lookback_minutes=60):
        """Analyze real-time trends for multiple symbols"""
        analysis = {}
        
        for symbol in symbols:
            try:
                # Get current real-time data
                current_data = self.get_realtime_data(symbol)
                
                if not current_data:
                    continue
                
                # Calculate trend analysis
                price = current_data['price']
                change_percent = current_data['change_percent']
                volume = current_data.get('volume', 0)
                
                # Determine trend
                if change_percent > 2:
                    trend = 'STRONG_BULLISH'
                    trend_emoji = '🚀'
                elif change_percent > 0.5:
                    trend = 'BULLISH'
                    trend_emoji = '📈'
                elif change_percent < -2:
                    trend = 'STRONG_BEARISH'
                    trend_emoji = '📉'
                elif change_percent < -0.5:
                    trend = 'BEARISH'
                    trend_emoji = '🔻'
                else:
                    trend = 'NEUTRAL'
                    trend_emoji = '➡️'
                
                # Volume analysis
                if volume > 0:
                    # This is simplified - in production you'd compare with average volume
                    volume_status = 'HIGH' if volume > 1000000 else 'NORMAL'
                else:
                    volume_status = 'UNKNOWN'
                
                analysis[symbol] = {
                    'current_price': price,
                    'change': current_data['change'],
                    'change_percent': change_percent,
                    'trend': trend,
                    'trend_emoji': trend_emoji,
                    'volume': volume,
                    'volume_status': volume_status,
                    'day_high': current_data.get('day_high', 0),
                    'day_low': current_data.get('day_low', 0),
                    'timestamp': current_data['timestamp'],
                    'source': current_data['source']
                }
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        return analysis
    
    def get_global_market_overview(self):
        """Get overview of global markets"""
        markets = {}
        
        for market, timezone in self.market_timezones.items():
            status = self.get_market_status(market)
            markets[market] = {
                'timezone': timezone,
                'status': status['status'],
                'local_time': status['local_time'],
                'reason': status['reason']
            }
        
        return markets
    
    def start_realtime_updates(self, symbols, update_interval=60):
        """Start real-time updates for given symbols"""
        self.is_running = True
        
        def update_loop():
            while self.is_running:
                try:
                    # Update market data
                    for symbol in symbols:
                        data = self.get_realtime_data(symbol)
                        if data:
                            self.market_data[symbol] = data
                            self.last_update[symbol] = datetime.now()
                    
                    # Sleep for update interval
                    time.sleep(update_interval)
                    
                except Exception as e:
                    print(f"Real-time update error: {str(e)}")
                    time.sleep(update_interval)
        
        # Start update thread
        update_thread = Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        return True
    
    def stop_realtime_updates(self):
        """Stop real-time updates"""
        self.is_running = False
    
    def get_market_movers(self, limit=10):
        """Get top market movers (gainers/losers)"""
        # This would typically fetch from a market data API
        # For now, return sample data structure
        return {
            'gainers': [
                {'symbol': 'NVDA', 'change_percent': 5.2, 'price': 450.00},
                {'symbol': 'AMD', 'change_percent': 3.8, 'price': 105.50},
                {'symbol': 'TSLA', 'change_percent': 2.9, 'price': 245.30}
            ],
            'losers': [
                {'symbol': 'META', 'change_percent': -2.1, 'price': 298.50},
                {'symbol': 'NFLX', 'change_percent': -1.8, 'price': 485.20},
                {'symbol': 'GOOGL', 'change_percent': -1.2, 'price': 2750.80}
            ]
        }
