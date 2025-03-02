import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import datetime as dt
from scipy.optimize import minimize
from fredapi import Fred
import ssl

class PortfolioOptimiser:  
   def __init__(self, tickers, start_date=None, end_date=None, fred_api_key="a889839f0ed2c5d5ea53eeef0c2b2541", min_allocations=None):
       # SSL handling
       try:
           _create_unverified_https_context = ssl._create_unverified_context
       except AttributeError:
           pass
       else:
           ssl._create_default_https_context = _create_unverified_https_context
      
       self.tickers = tickers
       self.end_date = end_date if end_date else dt.datetime.now()
       self.start_date = start_date if start_date else self.end_date - dt.timedelta(days=100)
       self.fred_api_key = fred_api_key
       
       # Set minimum allocations with default to 0
       self.min_allocations = {ticker: min_allocations.get(ticker, 0.0) if min_allocations else 0.0 
                               for ticker in tickers}
      
       # Initialise data structures
       self.adj_close_df = None
       self.log_returns = None
       self.cov_matrix = None
       self.correlation_matrix = None
       self.optimal_weights = None
       self.risk_free_rate = None
      
       # Fetch and process data
       self._fetch_data()
       self._calculate_returns()
       self._get_risk_free_rate()
  
 def _fetch_data(self):
    # Fetch historical price data for all tickers
    self.adj_close_df = pd.DataFrame()
    with st.spinner('Fetching market data...'):
        for ticker in self.tickers:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=False)  # updated line
            if len(data) > 0:
                self.adj_close_df[ticker] = data["Adj Close"]
            else:
                st.warning(f"No data found for ticker: {ticker}")
         
   def _calculate_returns(self):
       # Check for enough data
       if self.adj_close_df.empty or len(self.adj_close_df.columns) < 2:
           raise ValueError("Insufficient data to calculate returns. Please check your tickers.")
       
       self.log_returns = np.log(self.adj_close_df/self.adj_close_df.shift(1))
       self.log_returns = self.log_returns.dropna()
       self.cov_matrix = self.log_returns.cov() * 252
       self.correlation_matrix = self.log_returns.corr()
  
   def _get_risk_free_rate(self):
       fred = Fred(api_key=self.fred_api_key)
       risk_free_series = fred.get_series_latest_release("GS10")
       self.risk_free_rate = risk_free_series.iloc[-1]/100
  
   def _standard_deviation(self, weights):
       variance = weights.T @ self.cov_matrix @ weights
       return np.sqrt(variance)
  
   def _expected_return(self, weights):
       return np.sum(self.log_returns.mean() * weights) * 252
  
   def _sharpe_ratio(self, weights):
       return (self._expected_return(weights) - self.risk_free_rate) / self._standard_deviation(weights)
  
   def _neg_sharpe_ratio(self, weights):
       return -self._sharpe_ratio(weights)
  
   def optimise_portfolio(self): 
       # Constraints for minimum allocations and full investment
       def allocation_constraints(weights):
           # Check min allocations
           allocation_check = all(weights[i] >= self.min_allocations[self.tickers[i]] 
                                  for i in range(len(self.tickers)))
           # Check full investment
           full_investment = np.abs(np.sum(weights) - 1) < 1e-10
           return allocation_check and full_investment
       
       constraints = [
           {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
           {'type': 'ineq', 'fun': lambda weights: 
               [weights[i] - self.min_allocations[self.tickers[i]] for i in range(len(self.tickers))]
           }
       ]
       
       bounds = [(self.min_allocations.get(ticker, 0), 1) for ticker in self.tickers]
       
       # Adjust initial weights to respect minimum allocations
       initial_weights = np.array([max(self.min_allocations.get(ticker, 0.0), 0.01) for ticker in self.tickers])
       remaining_weight = 1.0 - np.sum(initial_weights)
       
       # Distribute remaining weight proportionally
       if remaining_weight > 0:
           prop_weights = np.array([w / np.sum(initial_weights) * remaining_weight if w > 0 else 0 for w in initial_weights])
           initial_weights += prop_weights
       
       # Normalise to ensure sum is 1
       initial_weights = initial_weights / np.sum(initial_weights)
      
       try:
           optimised_results = minimize(
               self._neg_sharpe_ratio,
               initial_weights,
               method='SLSQP',
               constraints=constraints,
               bounds=bounds
           )
      
           self.optimal_weights = optimised_results.x
           return self.get_portfolio_stats()
       except Exception as e:
           # If optimisation fails, use initial constrained weights
           self.optimal_weights = initial_weights
           st.warning(f"Optimisation failed. Using constrained initial weights. Error: {e}")
           return self.get_portfolio_stats()
  
   def get_portfolio_stats(self):
       if self.optimal_weights is None:
           raise ValueError("Portfolio must be optimised first")
      
       return {
           'weights': dict(zip(self.tickers, self.optimal_weights)),
           'expected_return': self._expected_return(self.optimal_weights),
           'volatility': self._standard_deviation(self.optimal_weights),
           'sharpe_ratio': self._sharpe_ratio(self.optimal_weights),
           'risk_free_rate': self.risk_free_rate,
           'correlation_matrix': self.correlation_matrix
       }
  
   def run_monte_carlo(self, initial_portfolio=100000, days=100, simulations=100):
       if self.optimal_weights is None:
           raise ValueError("Portfolio must be optimised first")
      
       mean_matrix = np.full(shape=(days, len(self.optimal_weights)),
                           fill_value=self.log_returns.mean()*252)
       mean_matrix = mean_matrix.T
       portfolio_sims = np.full(shape=(days, simulations), fill_value=0.0)
      
       with st.spinner('Running Monte Carlo simulation...'):
           for m in range(simulations):
               Z = np.random.normal(size=(days, len(self.optimal_weights)))
               L = np.linalg.cholesky(self.cov_matrix)
               daily_returns = mean_matrix/252 + np.inner(L/np.sqrt(252), Z)
               portfolio_sims[:, m] = initial_portfolio * np.exp(
                   np.cumsum(np.inner(self.optimal_weights, daily_returns.T))
               )
      
       mean_path = portfolio_sims.mean(axis=1)
       percentile_5 = np.percentile(portfolio_sims, 5, axis=1)
       percentile_95 = np.percentile(portfolio_sims, 95, axis=1)
      
       return {
           'final_mean': mean_path[-1],
           'final_5th_percentile': percentile_5[-1],
           'final_95th_percentile': percentile_95[-1],
           'simulation_paths': portfolio_sims,
           'mean_path': mean_path,
           'percentile_5': percentile_5,
           'percentile_95': percentile_95
       }


def main():
   st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
  
   st.title("SMF Bogle Fund Management Dashboard")

   st.subheader("Sharpe Ratio Maximiser")
  
   st.sidebar.header("Portfolio Settings")
  
   default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
  
   ticker_input = st.sidebar.text_area(
       "Enter stock tickers (one per line)",
       "\n".join(default_tickers)
   )
  
   # Convert text input to list and remove empty strings
   tickers = [ticker.strip() for ticker in ticker_input.split("\n") if ticker.strip()]
  
   end_date = st.sidebar.date_input("End Date", dt.datetime.now())
   start_date = st.sidebar.date_input("Start Date", end_date - dt.timedelta(days=100))
  
   # Monte Carlo settings
   initial_investment = st.sidebar.number_input("Initial Investment (€)", value=100000, step=10000)
   simulation_days = st.sidebar.number_input("Simulation Days", value=100, step=21)
   num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, step=100)
  
   try:
       # Prepare constraint dataframe
       constraint_df = pd.DataFrame({
           'Ticker': tickers,
           'Minimum Allocation (%)': [0.0] * len(tickers)
       })
      
       # First row: Portfolio Statistics and Allocation
       col1, col2 = st.columns(2)
       
       # Second row: Correlation and Constraints
       col3, col4 = st.columns(2)
      
       # Editable Minimum Allocation Constraints in second column
       with col4:
           st.subheader("Portfolio Constraints")
           edited_constraint_df = st.data_editor(
               constraint_df, 
               column_config={
                   'Minimum Allocation (%)': st.column_config.NumberColumn(
                       'Minimum Allocation (%)',
                       min_value=0.0,
                       max_value=100.0,
                       step=1.0
                   )
               },
               use_container_width=True,
               key="min_allocation_editor"
           )
       
       # Convert minimum allocations to dictionary
       min_allocations = dict(zip(
           edited_constraint_df['Ticker'], 
           edited_constraint_df['Minimum Allocation (%)'] / 100
       ))
      
       # Start optimiser with minimum allocations
       optimiser = PortfolioOptimiser(
           tickers, 
           start_date=start_date, 
           end_date=end_date, 
           min_allocations=min_allocations
       )
      
       # Run optimisation
       with st.spinner('Optimising portfolio...'):
           stats = optimiser.optimise_portfolio()
      
       # Portfolio Statistics in first column
       with col1:
           st.subheader("Portfolio Statistics")
           st.metric("Expected Annual Return", f"{stats['expected_return']:.2%}")
           st.metric("Expected Annual Volatility", f"{stats['volatility']:.2%}")
           st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
           st.metric("Risk-Free Rate", f"{stats['risk_free_rate']:.2%}")
      
       # Portfolio Allocation Pie Chart in second column
       with col2:
           filtered_weights = {k: v for k, v in stats['weights'].items() if v > 0.001}
           
           if filtered_weights:
               fig = px.pie(
                   values=list(filtered_weights.values()),
                   names=list(filtered_weights.keys()),
                   title="Optimal Portfolio Allocation"
               )
               st.plotly_chart(fig)
           else:
               st.warning("No significant allocations found. Please adjust your constraints.")
      
       # Correlation Matrix in first column of second row
       with col3:
           correlation_data = stats['correlation_matrix']
           
           # Create Plotly heatmap for correlation matrix
           heatmap_z = correlation_data.values
           x = list(correlation_data.columns)
           y = list(correlation_data.index)
           
           correlation_fig = ff.create_annotated_heatmap(
               z=heatmap_z, 
               x=x, 
               y=y,
               colorscale='Viridis',
               showscale=True,
               annotation_text=[[f'{val:.2f}' for val in row] for row in heatmap_z]
           )
           correlation_fig.update_layout(
               title="Correlation Between Selected Assets",
               height=500
           )
           st.plotly_chart(correlation_fig, use_container_width=True)
      
       # Monte Carlo Simulation
       st.subheader("Monte Carlo Simulation")
       sim_results = optimiser.run_monte_carlo(
           initial_portfolio=initial_investment,
           days=simulation_days,
           simulations=num_simulations
       )
      
       # Create Monte Carlo plot using plotly
       fig = go.Figure()
      
       # Add simulation paths with low opacity
       for i in range(sim_results['simulation_paths'].shape[1]):
           fig.add_trace(go.Scatter(
               y=sim_results['simulation_paths'][:, i],
               mode='lines',
               line=dict(color='white', width=0.5),
               opacity=0.1,
               showlegend=False
           ))
      
       # Add statistical lines
       fig.add_trace(go.Scatter(
           y=sim_results['mean_path'],
           mode='lines',
           name='Mean Path',
           line=dict(color='blue', width=2)
       ))
      
       fig.add_trace(go.Scatter(
           y=sim_results['percentile_5'],
           mode='lines',
           name='5th Percentile',
           line=dict(color='red', width=2, dash='dash')
       ))
      
       fig.add_trace(go.Scatter(
           y=sim_results['percentile_95'],
           mode='lines',
           name='95th Percentile',
           line=dict(color='green', width=2, dash='dash')
       ))
      
       fig.add_hline(
           y=initial_investment,
           line_dash="dash",
           line_color="black"
       )
      
       fig.update_layout(
           title="Monte Carlo Simulation of Optimised Portfolio Value",
           xaxis_title="Days",
           yaxis_title="Portfolio Value (€)",
           showlegend=True
       )
      
       st.plotly_chart(fig, use_container_width=True)
      
       # Display Monte Carlo statistics
       st.subheader("Monte Carlo Statistics")
       col5, col6, col7 = st.columns(3)
       col5.metric("Mean Final Value", f"€{sim_results['final_mean']:,.0f}")
       col6.metric("5th Percentile", f"€{sim_results['final_5th_percentile']:,.0f}")
       col7.metric("95th Percentile", f"€{sim_results['final_95th_percentile']:,.0f}")
      
   except Exception as e:
       st.error(f"An error occurred: {str(e)}")
       st.error("Please check your inputs and try again.")


if __name__ == "__main__":
   main()
