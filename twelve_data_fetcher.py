import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TwelveDataFetcher:
    """
    Twelve Data fetcher - reliable and generous free tier
    """
    
    def __init__(self):
        self.api_key = "5de724fc6c2d4363b02efeb9c0d12d0e"  # Your Twelve Data API key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_sp500_symbols(self):
        """Get S&P 500 symbols"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            return symbols
        except Exception as e:
            print(f"Error fetching S&P 500 symbols: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ']
    
    def fetch_stock_data(self, symbol, start_date=None, end_date=None, period='1y'):
        """Fetch stock data from Twelve Data"""
        print(f"🔄 Fetching REAL data for {symbol} from Twelve Data...")
        
        try:
            # Calculate date range
            if period == '1y':
                days = 365
            elif period == '6mo':
                days = 180
            elif period == '3mo':
                days = 90
            elif period == '1mo':
                days = 30
            else:
                days = 365
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for Twelve Data
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': '1day',
                'start_date': start_str,
                'end_date': end_str,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data_json = response.json()
                
                if 'values' in data_json and data_json['values']:
                    # Create DataFrame from Twelve Data
                    df = pd.DataFrame(data_json['values'])
                    
                    # Convert datetime column
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    
                    # Convert to numeric
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if not df.empty and len(df) >= 5:
                        print(f"✅ {symbol}: Got {len(df)} REAL data points from Twelve Data")
                        return df
                    else:
                        print(f"❌ {symbol}: Not enough data points from Twelve Data")
                        return None
                else:
                    print(f"❌ {symbol}: No data found in Twelve Data response")
                    return None
            else:
                print(f"❌ {symbol}: HTTP {response.status_code} from Twelve Data")
                if response.status_code == 429:
                    print("⚠️ Rate limit reached - waiting before retry...")
                    time.sleep(5)
                return None
                
        except Exception as e:
            print(f"❌ {symbol}: Error fetching from Twelve Data - {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols, start_date=None, end_date=None, period='1y'):
        """Fetch data for multiple stocks"""
        data_dict = {}
        successful_symbols = []
        
        print(f"🔄 Fetching REAL data for {len(symbols)} stocks from Twelve Data...")
        print(f"🔑 Using API key: {self.api_key[:8]}...")
        
        for i, symbol in enumerate(symbols):
            print(f"📊 Fetching {symbol} ({i+1}/{len(symbols)})")
            
            data = self.fetch_stock_data(symbol, start_date, end_date, period)
            
            if data is not None and not data.empty:
                data_dict[symbol] = data
                successful_symbols.append(symbol)
                print(f"✅ {symbol}: Success")
            else:
                print(f"❌ {symbol}: Failed")
            
            # Add delay between requests (Twelve Data allows 800 calls/day)
            if i < len(symbols) - 1:
                delay = random.uniform(1, 2)
                time.sleep(delay)
        
        if not data_dict:
            print("❌ No data fetched for any symbols")
            return None, []
        
        # Combine all data into a single DataFrame
        try:
            combined_data = pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())
            print(f"✅ Successfully fetched REAL data for {len(successful_symbols)} stocks")
            return combined_data, successful_symbols
        except Exception as e:
            print(f"Error combining data: {e}")
            return None, successful_symbols
    
    def get_quote(self, symbol):
        """Get real-time quote"""
        try:
            url = "https://api.twelvedata.com/quote"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data_json = response.json()
                
                if 'price' in data_json:
                    return {
                        'symbol': symbol,
                        'price': float(data_json['price']),
                        'change': float(data_json.get('change', 0)),
                        'change_percent': float(data_json.get('percent_change', 0)),
                        'high': float(data_json.get('high', 0)),
                        'low': float(data_json.get('low', 0)),
                        'open': float(data_json.get('open', 0)),
                        'volume': int(data_json.get('volume', 0))
                    }
            
            return None
            
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None

def get_sp500_symbols():
    """Standalone function to get S&P 500 symbols"""
    fetcher = TwelveDataFetcher()
    return fetcher.get_sp500_symbols()
