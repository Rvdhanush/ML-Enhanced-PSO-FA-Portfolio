import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AlphaVantageFetcher:
    """
    Alpha Vantage data fetcher using your free API key
    """
    
    def __init__(self):
        self.api_key = "HXR12B10KGHPU95P"  # Your dedicated API key
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
        """Fetch stock data using Alpha Vantage"""
        print(f"🔄 Fetching REAL data for {symbol} from Alpha Vantage...")
        
        try:
            # Add delay to respect rate limits
            time.sleep(random.uniform(1, 2))
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full' if period == '1y' else 'compact'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data_json = response.json()
                
                # Check for API limit message
                if 'Note' in data_json:
                    print(f"⚠️ API limit reached: {data_json['Note']}")
                    return None
                
                if 'Error Message' in data_json:
                    print(f"❌ Error: {data_json['Error Message']}")
                    return None
                
                if 'Time Series (Daily)' in data_json:
                    time_series = data_json['Time Series (Daily)']
                    
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    
                    # Rename columns
                    df = df.rename(columns={
                        '1. open': 'Open',
                        '2. high': 'High',
                        '3. low': 'Low',
                        '4. close': 'Close',
                        '5. volume': 'Volume'
                    })
                    
                    # Convert to numeric
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Filter by date range if specified
                    if start_date and end_date:
                        df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if not df.empty and len(df) >= 5:
                        print(f"✅ {symbol}: Got {len(df)} REAL data points from Alpha Vantage")
                        return df
                    else:
                        print(f"❌ {symbol}: Not enough data points")
                        return None
                else:
                    print(f"❌ {symbol}: No time series data found")
                    return None
            else:
                print(f"❌ {symbol}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols, start_date=None, end_date=None, period='1y'):
        """Fetch data for multiple stocks"""
        data_dict = {}
        successful_symbols = []
        
        print(f"🔄 Fetching REAL data for {len(symbols)} stocks from Alpha Vantage...")
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
            
            # Add delay between requests to respect rate limits
            if i < len(symbols) - 1:
                delay = random.uniform(2, 4)
                print(f"⏳ Waiting {delay:.1f}s to respect rate limits...")
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
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data_json = response.json()
                
                if 'Global Quote' in data_json:
                    quote = data_json['Global Quote']
                    return {
                        'symbol': quote['01. symbol'],
                        'price': float(quote['05. price']),
                        'change': float(quote['09. change']),
                        'change_percent': quote['10. change percent'].replace('%', ''),
                        'volume': int(quote['06. volume']),
                        'high': float(quote['03. high']),
                        'low': float(quote['04. low']),
                        'open': float(quote['02. open'])
                    }
            
            return None
            
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None

def get_sp500_symbols():
    """Standalone function to get S&P 500 symbols"""
    fetcher = AlphaVantageFetcher()
    return fetcher.get_sp500_symbols()
