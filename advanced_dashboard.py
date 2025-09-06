import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from realtime_market_analyzer import RealTimeMarketAnalyzer
from investment_calendar import InvestmentCalendar
from twelve_data_fetcher import TwelveDataFetcher
from accurate_ml_predictor import AccurateMLPredictor

# Initialize components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
realtime_analyzer = RealTimeMarketAnalyzer()
investment_calendar = InvestmentCalendar()
data_fetcher = TwelveDataFetcher()
ml_predictor = AccurateMLPredictor()

# Professional Layout like Groww/S&P Global
app.layout = dbc.Container([
    # Header with Real-Time Market Status
    dbc.Row([
        dbc.Col([
            html.H1("🌍 Global Stock Market Intelligence", className="text-center text-primary mb-2"),
            html.P("Real-Time Analysis • Investment Calendar • AI Predictions", 
                   className="text-center text-success mb-4"),
            
            # Real-Time Market Status Bar
            html.Div(id="market-status-bar", className="mb-4")
        ])
    ]),
    
    # Main Navigation Tabs
    dbc.Tabs([
        # Real-Time Analysis Tab
        dbc.Tab([
            html.Div(id="realtime-content")
        ], label="🔴 LIVE Market Analysis", tab_id="realtime-tab"),
        
        # Investment Calendar Tab  
        dbc.Tab([
            html.Div(id="calendar-content")
        ], label="📅 Investment Calendar", tab_id="calendar-tab"),
        
        # Portfolio Analysis Tab
        dbc.Tab([
            html.Div(id="portfolio-content")
        ], label="📊 Portfolio Analysis", tab_id="portfolio-tab"),
        
        # Global Markets Tab
        dbc.Tab([
            html.Div(id="global-markets-content")
        ], label="🌐 Global Markets", tab_id="global-tab")
        
    ], id="main-tabs", active_tab="realtime-tab"),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎛️ Control Panel", className="bg-info text-white"),
                dbc.CardBody([
                    # Stock Selection
                    html.Label("Select Stocks for Analysis:"),
                    dcc.Dropdown(
                        id='stock-selector',
                        options=[
                            {'label': '🍎 Apple (AAPL)', 'value': 'AAPL'},
                            {'label': '🪟 Microsoft (MSFT)', 'value': 'MSFT'},
                            {'label': '🔍 Google (GOOGL)', 'value': 'GOOGL'},
                            {'label': '⚡ Tesla (TSLA)', 'value': 'TSLA'},
                            {'label': '💎 NVIDIA (NVDA)', 'value': 'NVDA'},
                            {'label': '📘 Meta (META)', 'value': 'META'},
                            {'label': '📦 Amazon (AMZN)', 'value': 'AMZN'},
                            {'label': '🎬 Netflix (NFLX)', 'value': 'NFLX'}
                        ],
                        value=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
                        multi=True,
                        className='mb-3'
                    ),
                    
                    # Analysis Type
                    html.Label("Analysis Type:"),
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': '🔴 Real-Time (Live)', 'value': 'realtime'},
                            {'label': '📅 Weekly Calendar', 'value': 'weekly'},
                            {'label': '📅 Monthly Calendar', 'value': 'monthly'},
                            {'label': '🤖 AI Predictions', 'value': 'ai'}
                        ],
                        value='realtime',
                        className='mb-3'
                    ),
                    
                    # Update Button
                    dbc.Button(
                        "🔄 Update Analysis",
                        id='update-button',
                        color='primary',
                        size='lg',
                        className='w-100 mb-3'
                    ),
                    
                    # Auto-Refresh Toggle
                    dbc.Switch(
                        id="auto-refresh-switch",
                        label="🔄 Auto-Refresh (30s)",
                        value=False,
                        className='mb-3'
                    ),
                    
                    # Last Update Time
                    html.Small(id="last-update-time", className="text-muted")
                ])
            ])
        ], width=3),
        
        # Main Content Area
        dbc.Col([
            html.Div(id="main-content-area")
        ], width=9)
    ], className="mt-4"),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # 30 seconds
        n_intervals=0,
        disabled=True
    )
    
], fluid=True)

# Callback for Market Status Bar
@app.callback(
    Output('market-status-bar', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_market_status(n):
    """Update global market status bar"""
    try:
        global_markets = realtime_analyzer.get_global_market_overview()
        
        status_cards = []
        for market, info in global_markets.items():
            # Status color
            if info['status'] == 'OPEN':
                color = 'success'
                icon = '🟢'
            elif info['status'] == 'CLOSED':
                color = 'secondary'
                icon = '🔴'
            else:
                color = 'warning'
                icon = '🟡'
            
            card = dbc.Badge([
                f"{icon} {market}: {info['status']} ({info['local_time']})"
            ], color=color, className="me-2 mb-2")
            
            status_cards.append(card)
        
        return html.Div([
            html.H6("🌍 Global Market Status:", className="mb-2"),
            html.Div(status_cards)
        ])
        
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Market status update failed: {str(e)}", color="warning")
        ])

# Main Content Callback
@app.callback(
    Output('main-content-area', 'children'),
    [Input('update-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('stock-selector', 'value'),
     State('analysis-type', 'value'),
     State('auto-refresh-switch', 'value')]
)
def update_main_content(n_clicks, n_intervals, selected_stocks, analysis_type, auto_refresh):
    """Update main content area based on selected analysis type"""
    
    if not selected_stocks:
        return html.Div([
            dbc.Alert("Please select stocks to analyze", color="info")
        ])
    
    try:
        if analysis_type == 'realtime':
            return create_realtime_analysis(selected_stocks)
        elif analysis_type == 'weekly':
            return create_weekly_calendar(selected_stocks)
        elif analysis_type == 'monthly':
            return create_monthly_calendar(selected_stocks)
        elif analysis_type == 'ai':
            return create_ai_predictions(selected_stocks)
        else:
            return html.Div([
                dbc.Alert("Invalid analysis type", color="danger")
            ])
            
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])

def create_realtime_analysis(symbols):
    """Create real-time analysis content"""
    analysis = realtime_analyzer.analyze_realtime_trends(symbols)
    
    if not analysis:
        return html.Div([
            dbc.Alert("No real-time data available", color="warning")
        ])
    
    # Create cards for each symbol
    cards = []
    for symbol, data in analysis.items():
        # Determine card color based on trend
        if 'BULLISH' in data['trend']:
            card_color = 'success'
        elif 'BEARISH' in data['trend']:
            card_color = 'danger'
        else:
            card_color = 'info'
        
        card = dbc.Card([
            dbc.CardHeader([
                html.H5(f"{data['trend_emoji']} {symbol}", className="mb-0 text-white")
            ], className=f"bg-{card_color} text-white"),
            dbc.CardBody([
                html.H4(f"${data['current_price']:.2f}", className=f"text-{card_color}"),
                html.P([
                    html.Span(f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)", 
                             className=f"text-{card_color}"),
                    html.Br(),
                    html.Small(f"Trend: {data['trend']}", className="text-muted"),
                    html.Br(),
                    html.Small(f"Volume: {data['volume']:,}", className="text-muted"),
                    html.Br(),
                    html.Small(f"Range: ${data['day_low']:.2f} - ${data['day_high']:.2f}", className="text-muted")
                ]),
                html.Hr(),
                html.Small(f"Source: {data['source']} | {data['timestamp'][:19]}", className="text-muted")
            ])
        ], className="mb-3")
        
        cards.append(dbc.Col(card, width=6))
    
    return html.Div([
        html.H3("🔴 LIVE Market Analysis", className="text-danger mb-4"),
        html.P("Real-time data with automatic updates", className="text-muted mb-4"),
        dbc.Row(cards)
    ])

def create_weekly_calendar(symbols):
    """Create weekly investment calendar"""
    weekly_data = investment_calendar.generate_weekly_calendar(symbols, weeks_ahead=4)
    
    calendar_content = []
    
    for week_key, week_data in weekly_data.items():
        week_cards = []
        
        for symbol, recommendation in week_data['recommendations'].items():
            # Color based on recommendation
            if 'STRONG BUY' in recommendation['recommendation']:
                color = 'success'
            elif 'BUY' in recommendation['recommendation']:
                color = 'primary'
            elif 'HOLD' in recommendation['recommendation']:
                color = 'warning'
            else:
                color = 'danger'
            
            card = dbc.Card([
                dbc.CardHeader(f"{symbol} - Score: {recommendation['opportunity_score']}/100", 
                              className=f"bg-{color} text-white"),
                dbc.CardBody([
                    html.H6(recommendation['recommendation'], className=f"text-{color}"),
                    html.P(recommendation['action'], className="mb-2"),
                    html.Small([
                        f"Current: {recommendation['current_price']} | ",
                        f"Predicted: {recommendation['predicted_price']} | ",
                        f"RSI: {recommendation['rsi']}"
                    ], className="text-muted")
                ])
            ], className="mb-2")
            
            week_cards.append(dbc.Col(card, width=6))
        
        # Week section
        calendar_content.append(
            html.Div([
                html.H5(f"📅 {week_data['date_range']}", className="text-primary mb-3"),
                dbc.Row(week_cards),
                html.Hr()
            ])
        )
    
    return html.Div([
        html.H3("📅 Weekly Investment Calendar", className="text-primary mb-4"),
        html.P("AI-powered weekly investment recommendations", className="text-muted mb-4"),
        html.Div(calendar_content)
    ])

def create_monthly_calendar(symbols):
    """Create monthly investment calendar"""
    monthly_data = investment_calendar.generate_monthly_calendar(symbols, months_ahead=6)
    
    calendar_content = []
    
    for month_key, month_data in monthly_data.items():
        month_cards = []
        
        for symbol, recommendation in month_data['recommendations'].items():
            # Color based on recommendation
            if 'EXCELLENT' in recommendation['recommendation']:
                color = 'success'
            elif 'GOOD' in recommendation['recommendation']:
                color = 'primary'
            elif 'NEUTRAL' in recommendation['recommendation']:
                color = 'warning'
            else:
                color = 'danger'
            
            card = dbc.Card([
                dbc.CardHeader(f"{symbol} - Score: {recommendation['opportunity_score']}/100", 
                              className=f"bg-{color} text-white"),
                dbc.CardBody([
                    html.H6(recommendation['recommendation'], className=f"text-{color}"),
                    html.P(recommendation['action'], className="mb-2"),
                    html.Small([
                        f"Current: {recommendation['current_price']} | ",
                        f"Predicted: {recommendation['predicted_price']} | ",
                        f"Monthly Change: {recommendation['monthly_change']}"
                    ], className="text-muted"),
                    html.Br(),
                    html.Small(recommendation['seasonal_factors'], className="text-info")
                ])
            ], className="mb-2")
            
            month_cards.append(dbc.Col(card, width=6))
        
        # Month section
        calendar_content.append(
            html.Div([
                html.H5(f"📅 {month_data['month_name']}", className="text-primary mb-3"),
                dbc.Row(month_cards),
                html.Hr()
            ])
        )
    
    return html.Div([
        html.H3("📅 Monthly Investment Calendar", className="text-primary mb-4"),
        html.P("Long-term investment planning with seasonal analysis", className="text-muted mb-4"),
        html.Div(calendar_content)
    ])

def create_ai_predictions(symbols):
    """Create AI predictions content"""
    predictions = []
    
    for symbol in symbols:
        try:
            # Get data and train model
            data = data_fetcher.fetch_stock_data(symbol, period='1y')
            if data is not None and not data.empty:
                ml_predictor.train_model(symbol, data)
                recommendation = ml_predictor.get_investment_recommendation(symbol, data)
                
                predictions.append({
                    'symbol': symbol,
                    'data': recommendation
                })
        except Exception as e:
            print(f"Error predicting {symbol}: {str(e)}")
    
    if not predictions:
        return html.Div([
            dbc.Alert("No AI predictions available", color="warning")
        ])
    
    # Create prediction cards
    pred_cards = []
    for pred in predictions:
        symbol = pred['symbol']
        data = pred['data']
        
        card = dbc.Card([
            dbc.CardHeader(f"🤖 {symbol} AI Analysis", className="bg-info text-white"),
            dbc.CardBody([
                html.H5(data['recommendation'], className="text-primary"),
                html.P(data['reason'], className="mb-2"),
                html.Hr(),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Small("Current Price:", className="text-muted"),
                            html.H6(data['current_price'], className="text-primary")
                        ], width=3),
                        dbc.Col([
                            html.Small("Predicted Price:", className="text-muted"),
                            html.H6(data['predicted_price'], className="text-success")
                        ], width=3),
                        dbc.Col([
                            html.Small("RSI:", className="text-muted"),
                            html.H6(data['rsi'], className="text-warning")
                        ], width=3),
                        dbc.Col([
                            html.Small("Volatility:", className="text-muted"),
                            html.H6(data['volatility'], className="text-danger")
                        ], width=3)
                    ])
                ])
            ])
        ], className="mb-3")
        
        pred_cards.append(dbc.Col(card, width=6))
    
    return html.Div([
        html.H3("🤖 AI Predictions", className="text-info mb-4"),
        html.P("Machine Learning powered investment recommendations", className="text-muted mb-4"),
        dbc.Row(pred_cards)
    ])

# Auto-refresh callback
@app.callback(
    Output('interval-component', 'disabled'),
    [Input('auto-refresh-switch', 'value')]
)
def toggle_auto_refresh(auto_refresh):
    return not auto_refresh

# Last update time callback
@app.callback(
    Output('last-update-time', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_last_time(n):
    return f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

if __name__ == '__main__':
    app.run(debug=True, port=8053)
