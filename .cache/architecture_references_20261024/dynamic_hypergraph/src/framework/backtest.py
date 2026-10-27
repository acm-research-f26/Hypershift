import numpy as np
import pandas as pd

"""Top-K Daily Buy-Hold-Sell Strategy"""

class Backtest():
    def __init__(self, top_k=20):

        self.top_k = top_k
        self.performance = {}
        self.bt_long = 1.0
        self.sharpe_li = []
        self.cumulative_returns = [1.0] # For MDD calculation

    def calculate_mdd(self, cumulative_returns):
        """
        Calculate Maximum Drawdown (MDD) based on the formula:
        MDD = (Trough Value - Peak Value) / Peak Value
        
        Args:
            cumulative_returns (np.array): Array of cumulative returns over time.
            
        Returns:
            float: Maximum Drawdown (MDD) as a positive value.
        """
        # Track the peak value at each time step
        peak_value = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown at each time step
        drawdown = (cumulative_returns - peak_value) / peak_value
        
        # Return the maximum drawdown (in absolute value)
        mdd = np.min(drawdown) # This will be a negative value
        return abs(mdd) # Return the positive magnitude

    def backtesting(self, predictions, gts):
        """
        Evaluate returns, Sharpe ratio, and MDD for a specific phase where both predictions
        and ground truths are in DataFrame format.
        
        Args:
            predictions (pd.DataFrame): Predicted values for all stocks in the phase (index: date, columns: tickers).
            ground_truths (pd.DataFrame): Actual values for all stocks in the phase (index: date, columns: tickers).
            top_k (int): Number of top stocks to consider for evaluation.
            
        Returns:
            dict: Performance metrics including total return, Sharpe ratio, and MDD.
        """
        ground_truth = pd.concat([gts[key]['trend_return'] for key in gts.keys()], axis=1)
        ground_truth.columns = gts.keys()

        # Ensure predictions and ground_truths have the same structure
        assert predictions.shape == ground_truth.shape, "Shape mismatch between predictions and ground_truths"
        assert all(predictions.columns == ground_truth.columns), "Column mismatch between predictions and ground_truths"

        # Iterate over each row (date)
        for date in predictions.index:
            prediction_row = predictions.loc[date].values # Predicted values for the date
            ground_truth_row = ground_truth.loc[date].values # Actual values for the date

            # Top-K ranking for predictions
            rank_pre = np.argsort(prediction_row)
            top_k_indices = rank_pre[-self.top_k:] # Select indices of top-K predictions

            # Backtesting for top-k
            real_ret_rat_top_k = np.sum(ground_truth_row[top_k_indices]) / self.top_k
            self.bt_long += real_ret_rat_top_k
            self.sharpe_li.append(real_ret_rat_top_k)
            self.cumulative_returns.append(self.bt_long)

        # Total return
        self.performance['Return'] = self.bt_long - 1

        # Sharpe ratio
        self.sharpe_li = np.array(self.sharpe_li)
        self.performance['SR'] = (
            np.mean(self.sharpe_li) / np.std(self.sharpe_li)
        ) * 15.87 if len(self.sharpe_li) > 1 else None

        # Maximum Drawdown (MDD)
        self.cumulative_returns = np.array(self.cumulative_returns)
        self.performance['MDD'] = self.calculate_mdd(self.cumulative_returns)

        return self.performance
    
