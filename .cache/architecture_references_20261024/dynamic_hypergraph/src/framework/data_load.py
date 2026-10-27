import numpy as np
from utils.common import get_data_baseline
from utils.data_base import DataLoadBase
from snapshot import  HypergraphSnapshots
from scipy import sparse
from torch_geometric import utils
import pandas as pd
import os
import torch
from tqdm import tqdm
import pickle


class DataLoad(DataLoadBase):
    def __init__(self, config):
        self.config = config
        self.symbols = config["data"]["symbols"]
        self.his_window = config['data']['history_window']
        self.indicators = config['data']['indicators']
        self.include_target = config["data"]["include_target"]
        self.target_col = config["data"]["target_col"]
        self.max_slow_period = max([max(indi.values()) for indi in self.indicators.values() if indi.values()])
        self.n_step_ahead = config['data']['n_step_ahead']
        self.outlier_threshold = config["data"]["outlier_threshold"]
        self.data_path = config['data']['data_path']
        self.cuda = config['model']['cuda']
        self.start_train = config['data']['start_train']

    def get_data(self, 
                 start_train, 
                 end_train, 
                 start_test, 
                 end_test):
        path_data = f"{self.data_path}/train/train_{start_train}.npz"
        path_hyper = f"./{self.data_path}/hypergraph/hypergraphsnapshot_{start_train}.pkl"
        if not os.path.exists(path_data):
            X_train_full, y_train_full, X_test_full, y_test_full, hypergraphsnapshot, edges, buy_prob_threshold, sell_prob_threshold = self.split_train_test(start_train, end_train, start_test, end_test)
            np.savez(path_data, x_train=X_train_full, y_train=y_train_full, x_test=X_test_full, y_test=y_test_full, edges=edges, buy_thres=buy_prob_threshold, sell_thres=sell_prob_threshold)
            if not os.path.exists(path_hyper):
                with open(path_hyper, "wb") as f:
                    pickle.dump(hypergraphsnapshot, f)
            else:
                with open(path_hyper, "rb") as f:
                    hypergraphsnapshot = pickle.load(f)
        else:
            print("Load the saved data!")
            data = np.load(path_data)
            X_train_full = data["x_train"]
            y_train_full = data["y_train"]
            X_test_full = data["x_test"]
            y_test_full = data["y_test"]
            edges = data["edges"]
            buy_prob_threshold = data["buy_thres"]
            sell_prob_threshold = data["sell_thres"]
            # train_data_storage = {}
            # for idx, sym in enumerate(self.symbols):
            #     train_data = get_data_baseline(sym, start_train, end_train, self.his_window + self.max_slow_period, self.n_step_ahead)
            #     if idx == 0:
            #         max_train_trade_index = train_data.index
            #     train_data = self.fill_missing_data(train_data, max_train_trade_index)
            #     train_data_storage[sym] = train_data
            # hypergraphsnapshot = HypergraphSnapshots(self.symbols, self.start_train, train_data_storage, self.cuda)
            with open(path_hyper, "rb") as f:
                hypergraphsnapshot = pickle.load(f)
            X_train_full = torch.tensor(X_train_full, dtype=torch.float32)
            y_train_full = torch.tensor(y_train_full, dtype=torch.float32)
            X_test_full = torch.tensor(X_test_full, dtype=torch.float32)
            y_test_full = torch.tensor(y_test_full, dtype=torch.float32)
        return X_train_full, y_train_full[:,:,-1], X_test_full, y_test_full[:,:,-1], hypergraphsnapshot, torch.LongTensor(edges), buy_prob_threshold, sell_prob_threshold

    def split_train_test(self,
                          start_train, 
                          end_train,
                          start_test,
                          end_test):
        
        # X_train_storage = []
        # X_test_storage = []
        # y_train_storage = []
        # y_test_storage = []
        train_data_storage = {}
        test_data_storage = {}
        buy_prob_threshold = []
        sell_prob_threshold = []

        for idx, sym in tqdm(enumerate(self.symbols)):

            train_data = get_data_baseline(sym, start_train, end_train, self.his_window + self.max_slow_period, self.n_step_ahead)
            test_data = get_data_baseline(sym, start_test, end_test, self.his_window + self.max_slow_period, self.n_step_ahead)

            if idx == 0:
                max_train_trade_index = train_data.index
                max_test_trade_index = test_data.index

            train_data = self.fill_missing_data(train_data, max_train_trade_index)
            test_data = self.fill_missing_data(test_data, max_test_trade_index)

            train_data_storage[sym] = train_data
            test_data_storage[sym] = test_data

            _, df_x_train, df_y_train = self.preprocess_data(train_data)
            _, df_x_test, df_y_test = self.preprocess_data(test_data)

            buy_prob_threshold.append(df_y_train.mean())
            sell_prob_threshold.append(-df_y_train.mean())

            train_x = df_x_train.to_numpy()
            train_y = df_y_train.to_numpy()
            test_x = df_x_test.to_numpy()
            test_y = df_y_test.to_numpy()

            if idx == 0:
                X_train = np.zeros((len(train_x)-self.his_window+1, len(self.symbols), self.his_window, train_x.shape[-1]))
                y_train = np.zeros((len(train_y)-self.his_window+1, len(self.symbols), self.his_window))
                X_test = np.zeros((len(test_x)-self.his_window+1, len(self.symbols), self.his_window, test_x.shape[-1]))
                y_test =  np.zeros((len(test_y)-self.his_window+1, len(self.symbols), self.his_window))

            X_train[:,idx,:] = np.array([train_x[i: i + self.his_window]
                                for i in range(len(train_x)-self.his_window+1)])
            y_train[:,idx,:] = np.array([train_y[i: i + self.his_window]
                                for i in range(len(train_y)-self.his_window+1)])
            X_test[:,idx,:] = np.array([test_x[i: i + self.his_window]
                            for i in range(len(test_x)-self.his_window+1)])
            y_test[:,idx,:] = np.array([test_y[i: i + self.his_window]
                            for i in range(len(test_y)-self.his_window+1)])
            
        #     X_train_storage.append(X_train)
        #     X_test_storage.append(X_test)
        #     y_train_storage.append(y_train)
        #     y_test_storage.append(y_test)
            
        # X_train_full = np.stack((X_train_storage), axis=1)
        # y_train_full = np.stack((y_train_storage), axis = 1)
        # X_test_full = np.stack((X_test_storage), axis=1)
        # y_test_full =  np.stack((y_test_storage), axis = 1)
            
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
                
        stock_list = pd.read_csv("../data/sp500/sp500_ticker.csv", index_col = "Symbol")
        cat_list = stock_list.loc[self.symbols]["Sector"].unique()
        cat_dict = {}
        for i in range(len(cat_list)):
            cat = cat_list[i]
            cat_dict[cat] = i
            
        incidence_matrix = np.zeros((len(self.symbols), len(cat_list)))
        for i in range(len(self.symbols)):
            cat_key = stock_list.loc[self.symbols[i]].Sector    
            cat_index = cat_dict[cat_key]
            incidence_matrix[i][cat_index] = 1
            
        inci_sparse = sparse.coo_matrix(incidence_matrix)
        incidence_edges = utils.from_scipy_sparse_matrix(inci_sparse)
        hypergraphsnapshot = HypergraphSnapshots(self.symbols, start_train, train_data_storage, self.cuda)
        print("Done!")

        return X_train, y_train, X_test, y_test, hypergraphsnapshot.hypergraph_snapshot, incidence_edges[0], buy_prob_threshold, sell_prob_threshold