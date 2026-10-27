import numpy as np
from utils.common import get_data_baseline
import utils.indicators as indi
import os

class DataLoadBase():
    def __init__(self, config):
        self.config = config
        self.symbols = config["data"]["symbols"]
        self.his_window = config["data"]["history_window"]
        self.indicators = config["data"]["indicators"]
        self.include_target = config["data"]["include_target"]
        self.target_col = config["data"]["target_col"]
        self.max_slow_period = max([max(indi.values()) for indi in self.indicators.values() if indi.values()])
        self.n_step_ahead = config["data"]["n_step_ahead"]
        self.outlier_threshold = config["data"]["outlier_threshold"]
        self.data_path = config['data']['data_path']

    def fill_missing_data(self, df, max_trade_index):

        missing_points = np.setdiff1d(max_trade_index, df.index)
        df.fillna(np.NaN, inplace=True)
        df = df.interpolate(limit_direction = "both")
        
        if len(missing_points) > 0:
            
            for ind in missing_points:
                df.loc[ind] = np.nan
            df = df.sort_index(axis = 0, ascending = True)
            df["open"] = df["open"].interpolate(limit_direction = "both")
            df["high"] = df["high"].interpolate(limit_direction = "both")
            df["low"] = df["low"].interpolate(limit_direction = "both")
            df["close"] = df["close"].interpolate(limit_direction = "both")
            df["volume"] = df["volume"].interpolate(limit_direction = "both")
            
        return df

    def preprocess_data(self, source_df):

        df = source_df.copy()
        features = []

        if "ohlcv_ratio" in self.indicators:
            period = self.indicators["ohlcv_ratio"]["period"]
            df, features = indi.ohlcv_ratio(df, features, period)

        if "close_ratio" in self.indicators:
            medium_period = self.indicators["close_ratio"]["medium_period"]
            slow_period = self.indicators["close_ratio"]["slow_period"]
            df, features = indi.close_ratio(df, features, [medium_period, slow_period])

        if "volume_ratio" in self.indicators:
            medium_period = self.indicators["volume_ratio"]["medium_period"]
            slow_period = self.indicators["volume_ratio"]["slow_period"]
            df, features = indi.volume_ratio(df, features, [medium_period, slow_period])

        if "close_sma" in self.indicators:
            medium_period = self.indicators["close_sma"]["medium_period"]
            slow_period = self.indicators["close_sma"]["slow_period"]
            df, features = indi.close_sma(df, features, [medium_period, slow_period])

        if "volume_sma" in self.indicators:
            medium_period = self.indicators["volume_sma"]["medium_period"]
            slow_period = self.indicators["volume_sma"]["slow_period"]
            df, features = indi.volume_sma(df, features, [medium_period, slow_period])
        
        if "close_ema" in self.indicators:
            medium_period = self.indicators["close_ema"]["medium_period"]
            slow_period = self.indicators["close_ema"]["slow_period"]
            df, features = indi.close_sma(df, features, [medium_period, slow_period])

        if "volume_ema" in self.indicators:
            medium_period = self.indicators["volume_ema"]["medium_period"]
            slow_period = self.indicators["volume_ema"]["slow_period"]
            df, features = indi.volume_sma(df, features, [medium_period, slow_period])

        if "atr" in self.indicators:
            medium_period = self.indicators["atr"]["medium_period"]
            slow_period = self.indicators["atr"]["slow_period"]
            df, features = indi.atr(df, features, [medium_period, slow_period])

        if "adx" in self.indicators:
            medium_period = self.indicators["adx"]["medium_period"]
            slow_period = self.indicators["adx"]["slow_period"]
            df, features = indi.adx(df, features, [medium_period, slow_period])
        
        if "kdj" in self.indicators:    
            medium_period = self.indicators["kdj"]["medium_period"]
            slow_period = self.indicators["kdj"]["slow_period"]
            df, features = indi.kdj(df, features, [medium_period, slow_period])

        if "rsi" in self.indicators:  
            medium_period = self.indicators["rsi"]["medium_period"]
            slow_period = self.indicators["rsi"]["slow_period"]          
            df, features = indi.rsi(df, features, [medium_period, slow_period])

        if "macd" in self.indicators:    
            medium_period = self.indicators["macd"]["medium_period"]
            slow_period = self.indicators["macd"]["slow_period"]        
            df, features = indi.macd(df, features, 9, 12, 26)
        
        if "mfi" in self.indicators:
            medium_period = self.indicators["mfi"]["medium_period"]
            slow_period = self.indicators["mfi"]["slow_period"]
            df, features = indi.mfi(df, features, [medium_period, slow_period])

        if "bb" in self.indicators:                
            df, features = indi.bb(df, features)

        if "arithmetic_returns" in self.indicators:
            df, features = indi.arithmetic_returns(df, features)
        
        if "obv" in self.indicators:
            medium_period = self.indicators["rsi"]["medium_period"]
            slow_period = self.indicators["rsi"]["slow_period"]
            df, features = indi.obv(df, features, [medium_period, slow_period])

        if "ichimoku" in self.indicators:
            fast_period = self.indicators["ichimoku"]["fast_period"]
            medium_period = self.indicators["ichimoku"]["medium_period"]
            slow_period = self.indicators["ichimoku"]["slow_period"]        
            df, features = indi.ichimoku(df, features, fast_period, medium_period, slow_period)
        
        if "k_line" in self.indicators:
            df, features = indi.k_line(df, features)
        
        if "eight_trigrams" in self.indicators:
            df, features = indi.eight_trigrams(df, features)

        if "trend_return" in self.indicators:
            df, features = indi.trend_return(df, features, self.n_step_ahead)

        df = df[self.max_slow_period:-self.n_step_ahead]
        if not self.include_target:
            features.remove(self.target_col)
        df = df.interpolate(limit_direction="both")
        df = indi.remove_outliers(df, features, threshold=self.outlier_threshold)
        df_y = df[self.target_col]
        df_x = df.filter((features))

        return df, df_x, df_y
    
    def gen_backtest_data(self, start_date, end_date):
        gts = {}
        inputs = []
        # inputs_storage = {}
        for idx, sym in enumerate(self.symbols):
            data = get_data_baseline(sym, start_date, end_date, self.his_window + self.max_slow_period + 5, self.n_step_ahead)
            # inputs_storage[sym] = data

            if idx == 0:
                max_data_trade_index = data.index
            data = self.fill_missing_data(data, max_data_trade_index)
        
            df, df_x, _ = self.preprocess_data(data)
            df = df.iloc[self.his_window:]
            df_x = df_x.to_numpy()
            gts[sym] = df[["open", "high", "low", "close", "volume", "trend_return"]]
            input_data = np.array([df_x[i: i + self.his_window] for i in range(len(df_x) - self.his_window)])
            inputs.append(input_data)
        num_sample = inputs[0].shape[0]
        inputs = np.vstack((inputs))
        inputs = np.reshape(inputs, (num_sample, len(self.symbols), inputs.shape[1], inputs.shape[2]))
        return inputs, gts
