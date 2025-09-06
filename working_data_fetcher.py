import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WorkingDataFetcher:
    """
    Working data fetcher that actually gets real data
    """
    
    def __init__(self):
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
        """Fetch stock data using working methods"""
        print(f"🔄 Fetching REAL data for {symbol}...")
        
        # Method 1: Try yfinance with different approach
        data = self._fetch_yfinance_working(symbol, start_date, end_date, period)
        if data is not None and not data.empty:
            return data
        
        # Method 2: Try direct CSV download
        data = self._fetch_csv_download(symbol, start_date, end_date, period)
        if data is not None and not data.empty:
            return data
        
        print(f"❌ All methods failed for {symbol}")
        return None
    
    def _fetch_yfinance_working(self, symbol, start_date=None, end_date=None, period='1y'):
        """Working yfinance method"""
        try:
            # Add delay
            time.sleep(random.uniform(1, 2))
            
            # Create ticker
            ticker = yf.Ticker(symbol)
            
            # Try different approaches
            approaches = [
                # Try with different periods
                lambda: ticker.history(period='1y', interval='1d', auto_adjust=True),
                lambda: ticker.history(period='6mo', interval='1d', auto_adjust=True),
                lambda: ticker.history(period='3mo', interval='1d', auto_adjust=True),
                lambda: ticker.history(period='1mo', interval='1d', auto_adjust=True),
                # Try with different intervals
                lambda: ticker.history(period='1y', interval='1d', auto_adjust=False),
                lambda: ticker.history(period='6mo', interval='1d', auto_adjust=False),
                # Try with prepost
                lambda: ticker.history(period='1y', interval='1d', prepost=True),
                lambda: ticker.history(period='6mo', interval='1d', prepost=True),
            ]
            
            for i, approach in enumerate(approaches):
                try:
                    data = approach()
                    if not data.empty and len(data) >= 5:
                        print(f"✅ {symbol}: Got {len(data)} REAL data points from yfinance_working (method {i+1})")
                        return data
                except Exception as e:
                    print(f"⚠️ yfinance_working method {i+1} failed: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ yfinance_working error: {str(e)}")
            return None
    
    def _fetch_csv_download(self, symbol, start_date=None, end_date=None, period='1y'):
        """Direct CSV download from Yahoo Finance"""
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
            
            # Convert to timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Try different Yahoo Finance URLs
            urls = [
                f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history",
                f"https://query2.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history",
            ]
            
            for i, url in enumerate(urls):
                try:
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        # Try to parse as CSV
                        try:
                            from io import StringIO
                            df = pd.read_csv(StringIO(response.text))
                            
                            if not df.empty and 'Date' in df.columns:
                                df['Date'] = pd.to_datetime(df['Date'])
                                df.set_index('Date', inplace=True)
                                df.sort_index(inplace=True)
                                
                                if not df.empty and len(df) >= 5:
                                    print(f"✅ {symbol}: Got {len(df)} REAL data points from CSV download (URL {i+1})")
                                    return df
                        except Exception as e:
                            print(f"⚠️ CSV parsing failed for URL {i+1}: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"⚠️ CSV download URL {i+1} failed: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ CSV download error: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols, start_date=None, end_date=None, period='1y'):
        """Fetch data for multiple stocks"""
        data_dict = {}
        successful_symbols = []
        
        print(f"🔄 Fetching REAL data for {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols):
            print(f"📊 Fetching {symbol} ({i+1}/{len(symbols)})")
            
            data = self.fetch_stock_data(symbol, start_date, end_date, period)
            
            if data is not None and not data.empty:
                data_dict[symbol] = data
                successful_symbols.append(symbol)
                print(f"✅ {symbol}: Success")
            else:
                print(f"❌ {symbol}: Failed")
        
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

def get_sp500_symbols():
    """Standalone function to get S&P 500 symbols"""
    fetcher = WorkingDataFetcher()
    return fetcher.get_sp500_symbols()
