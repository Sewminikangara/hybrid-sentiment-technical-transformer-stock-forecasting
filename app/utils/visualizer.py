"""
Chart Visualizer for Stock Prediction App
Creates interactive Plotly charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

class ChartVisualizer:
    """Create interactive charts"""
    
    def __init__(self):
        self.colors = {
            'actual': '#2E86AB',
            'predicted': '#A23B72',
            'upper': '#F18F01',
            'lower': '#F18F01',
            'buy': '#4CAF50',
            'sell': '#f44336'
        }
    
    def create_prediction_chart(self, historical_data, predictions, stock):
        """Create prediction chart with historical and predicted prices"""
        
        # Prepare historical data (last 60 days)
        hist = historical_data.tail(60).copy()
        hist_dates = pd.to_datetime(hist['Date'])
        hist_prices = hist['Close']
        
        # Create figure
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_prices,
            mode='lines',
            name='Historical',
            line=dict(color=self.colors['actual'], width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Predicted prices
        pred_dates = predictions['dates']
        pred_prices = predictions['prices']
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            name='Predicted',
            line=dict(color=self.colors['predicted'], width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1],
            y=predictions['upper'] + predictions['lower'][::-1],
            fill='toself',
            fillcolor='rgba(241, 143, 1, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{stock} Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_historical_chart(self, data, stock):
        """Create historical price chart with volume"""
        
        dates = pd.to_datetime(data['Date'])
        prices = data['Close']
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Close Price',
            line=dict(color=self.colors['actual'], width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)',
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{stock} - 60 Day History',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            showlegend=False,
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_comparison_chart(self, results, metric='MAPE'):
        """Create model comparison chart"""
        
        fig = px.bar(
            results,
            x='Model',
            y=metric,
            color='Model',
            title=f'Model Comparison - {metric}',
            template='plotly_dark'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        return fig
    
    def create_sentiment_chart(self, data):
        """Create sentiment timeline chart"""
        
        dates = pd.to_datetime(data['Date'])
        sentiment = data['compound']
        
        colors = ['green' if s > 0 else 'red' for s in sentiment]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dates,
            y=sentiment,
            marker_color=colors,
            name='Sentiment',
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sentiment:</b> %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Sentiment Analysis Timeline',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            template='plotly_dark',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_technical_indicators_chart(self, data):
        """Create technical indicators chart"""
        
        dates = pd.to_datetime(data['Date'])
        
        fig = go.Figure()
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=dates,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#FF6B6B', width=2)
            ))
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        
        fig.update_layout(
            title='Technical Indicators - RSI',
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_dark',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
