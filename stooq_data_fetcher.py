import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StooqDataFetcher:
    """
    Stooq data fetcher - completely free and reliable
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
        """Fetch stock data from Stooq"""
        print(f"🔄 Fetching REAL data for {symbol} from Stooq...")
        
        try:
            # Stooq uses different symbol format
            stooq_symbol = symbol.replace('.', '')
            
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
            
            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    
                    if not df.empty and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Filter by date range if specified
                        if start_date and end_date:
                            df = df[(df.index >= start_date) & (df.index <= end_date)]
                        
                        # Rename columns to match expected format
                        df = df.rename(columns={
                            'Open': 'Open',
                            'High': 'High',
                            'Low': 'Low',
                            'Close': 'Close',
                            'Volume': 'Volume'
                        })
                        
                        if not df.empty and len(df) >= 5:
                            print(f"✅ {symbol}: Got {len(df)} REAL data points from Stooq")
                            return df
                        else:
                            print(f"❌ {symbol}: Not enough data points from Stooq")
                            return None
                    else:
                        print(f"❌ {symbol}: No data columns found in Stooq response")
                        return None
                except Exception as e:
                    print(f"❌ {symbol}: Error parsing Stooq CSV - {str(e)}")
                    return None
            else:
                print(f"❌ {symbol}: HTTP {response.status_code} from Stooq")
                return None
                
        except Exception as e:
            print(f"❌ {symbol}: Error fetching from Stooq - {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols, start_date=None, end_date=None, period='1y'):
        """Fetch data for multiple stocks"""
        data_dict = {}
        successful_symbols = []
        
        print(f"🔄 Fetching REAL data for {len(symbols)} stocks from Stooq...")
        
        for i, symbol in enumerate(symbols):
            print(f"📊 Fetching {symbol} ({i+1}/{len(symbols)})")
            
            data = self.fetch_stock_data(symbol, start_date, end_date, period)
            
            if data is not None and not data.empty:
                data_dict[symbol] = data
                successful_symbols.append(symbol)
                print(f"✅ {symbol}: Success")
            else:
                print(f"❌ {symbol}: Failed")
            
            # Add delay between requests
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

def get_sp500_symbols():
    """Standalone function to get S&P 500 symbols"""
    fetcher = StooqDataFetcher()
    return fetcher.get_sp500_symbols()
