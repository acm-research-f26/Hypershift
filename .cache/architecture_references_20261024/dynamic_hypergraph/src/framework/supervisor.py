import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from utils.metric import RMSELoss, eval_all_metrics
from utils.common import add_month_to_string_time
# from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
import wandb
import os 
import copy
from data_load import DataLoad
from torch.utils.data import Dataset, DataLoader
from backtest import Backtest


def get_tickers_sp500():
    data = np.load(f'../data/sp500/baseline_data_sp500.npy', allow_pickle=True).item()
    tickers = data.keys()

    not_na_tickers = []
    for ticker in tickers:
        if data[ticker][(data[ticker].index >= '2012-11-01')&(data[ticker].index <= '2022-01-01')].isna().sum().sum() == 0:
            not_na_tickers.append(ticker)
    return not_na_tickers
    

class TSDataset(Dataset):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Supervisor():
    def __init__(self, config, device):

        self.model_instance = None

        self.config = config
        self.symbols = get_tickers_sp500()
        self.num_stocks = len(self.symbols)

        # Backtest
        self.backtest_config = config['backtest']
        self.res_path = self.backtest_config['res_path']
        os.makedirs(self.res_path, exist_ok=True)
        self.bt_top_k = self.backtest_config['top_k']
        self.start_backtest = self.backtest_config['start_backtest']
        self.backtest_month = self.backtest_config['backtest_month']
        self.end_backtest_final = self.backtest_config['end_backtest_final']
        self.end_backtest = add_month_to_string_time(self.start_backtest, self.backtest_month)

        # Train
        self.epochs = config['train']['epochs']
        self.learning_rate = config['train']['learning_rate']
        self.seed = config['train']['seed']
        self.batch_size = config['train']['batch_size']
        self.save_path = config['train']['save_path']
        os.makedirs(self.save_path, exist_ok=True)
        self.confidence_threshold = config['train']['confidence_threshold']
        self.early_stop = config['train']['early_stop']
        self.device = device
        
        # Data
        self.start_train = config["data"]["start_train"]
        self.start_test = config["data"]["start_test"]
        self.end_train = self.start_test
        self.end_test = self.start_backtest
        
        self.model = None
        self.optimizer = None
        self.run_name = config['run_name']

    def set_model_instanece(self, model_instance):

        self.model_instance = model_instance

    def init_model(self):

        if self.model is None:
            raise ValueError("Model has not been initialized")

        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.model.to(self.device)
        self.criterion = RMSELoss()

    def train(self):

        print(f"Model Initialization (Phase {self.phase})")

        self.data_loader = DataLoad(self.config)
        self.X_train, self.y_train, self.X_test, self.y_test, self.hypergraphsnapshot, _, _, _ = self.data_loader.get_data(self.start_train, self.end_train, self.start_test, self.end_test)
        self.model_instance._snapshot(self.hypergraphsnapshot)
        train_dataset = TSDataset(self.X_train, self.y_train)
        test_dataset = TSDataset(self.X_test, self.y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
       
        print("Begin training...")
        
        best_acc = -9999
        
        for epoch in tqdm(range(self.epochs), desc='epoch'):

            self.epoch = epoch

            self.model.train()

            train_loss = []

            dataset_len = len(train_loader.dataset)

            trues = np.zeros((dataset_len, self.num_stocks), dtype=np.float32)
            preds = np.zeros((dataset_len, self.num_stocks), dtype=np.float32)

            for idx, (input, true) in enumerate(train_loader):

                input = input.to(self.device) # (1, S, T, D)
                true = true.to(self.device) # (1, S)

                input = input.squeeze(0) # (S, T, D)

                self.optimizer.zero_grad()

                pred = self.model(input)
                pred = pred.unsqueeze(0) # (1, S)

                loss = self.criterion(true, pred)
                
                loss.backward()

                # clip_grad_value_(self.model.parameters(), 3.0) 

                self.optimizer.step()

                if true.shape[0] == self.batch_size:
                    weighted_loss = loss.detach().cpu().numpy()
                else:
                    weighted_loss = loss.detach().cpu().numpy() * self.batch_size / true.shape[0]

                train_loss.append(weighted_loss)

                trues[idx*self.batch_size:(idx+1)*self.batch_size,:] = true.detach().cpu().numpy()
                preds[idx*self.batch_size:(idx+1)*self.batch_size,:] = pred.detach().cpu().numpy()

            self.train_loss = sum(train_loss) / len(train_loss)
            train_metrics = eval_all_metrics(trues, preds)
            self.train_acc = train_metrics['acc']

            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch":self.epoch,
                        f"Train Loss (Phase {self.phase})":self.train_loss,
                        f"Train ACC (Phase {self.phase})":self.train_acc,
                    }
                )

            self.model.eval()
            with torch.no_grad():
                test_loss, test_metrics, test_gt = self.eval(test_loader)
                test_acc = test_metrics['acc']                
        
            if test_acc > best_acc and self.train_acc > 0.5:
                best_acc = test_acc
                best_param = copy.deepcopy(self.model.state_dict())
                self.save_path__ = self.save_path + self.run_name + f"_phase{self.phase}.pth"
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(best_param, self.save_path__)

            if self.train_acc >= self.confidence_threshold:
                confidence += 1
            else:
                confidence = 0
            if confidence >= self.early_stop:
                print("Early stop due to train acc reach")
                break

        print("Done training!")


    def test(self):

        print("Begin testing...")

        self.model.load_state_dict(torch.load(self.save_path__))
        self.model.eval()

        with torch.inference_mode():
            # Accuracy
            # test_preds = np.zeros(shape=(len(self.X_test), self.num_stocks))
            test_preds = np.zeros(shape=self.y_test.shape)
            for i in range(0, len(self.X_test), self.batch_size):
                test_preds[i:i+self.batch_size] = self.model(self.X_test[i:i+self.batch_size].squeeze(0).to(self.device)).cpu().detach().numpy()
            # remainder = len(self.X_test) % self.batch_size
            # test_preds[-remainder:] = self.model(self.X_test[-remainder:]).cpu().detach().numpy()
            dict_metrics = eval_all_metrics(self.y_test.cpu().detach().numpy(), test_preds)
            print("Results are saved in the log file!")

            # Backtest
            print("Backtesting...")
            input_backtest, gt_backtest = self.data_loader.gen_backtest_data(
                self.start_backtest, self.end_backtest)
            input_backtest = torch.tensor(input_backtest, dtype=torch.float32)
            input_backtest = input_backtest.to(self.device)
            preds = np.zeros(shape=(len(input_backtest), self.num_stocks))
            for i in range(0, len(input_backtest), self.batch_size):
                preds[i:i+self.batch_size] = self.model(input_backtest[i:i+self.batch_size].squeeze(0).to(self.device)).cpu().detach().numpy()
            # remainder = len(input_backtest) % self.batch_size
            # preds[-remainder:] = self.model(input_backtest[-remainder:]).cpu().detach().numpy()
    
            preds = pd.DataFrame(preds, columns=gt_backtest.keys(), index=gt_backtest[list(gt_backtest.keys())[0]].index)
            bt = Backtest(top_k=self.bt_top_k)
            final_report = bt.backtesting(preds, gt_backtest)
            print("Finish!")

        return final_report, dict_metrics


    def eval(self, data_loader):

        self.model.eval()

        test_loss = []

        dataset_len = len(data_loader.dataset)

        trues = np.zeros((dataset_len, self.num_stocks), dtype=np.float32)
        preds = np.zeros((dataset_len, self.num_stocks), dtype=np.float32)

        with torch.no_grad():
            for idx, (input, true) in enumerate(data_loader):

                input = input.to(self.device) # (1, S, T, D)
                true = true.to(self.device) # (1, S)

                input = input.squeeze(0) # (S, T, D)

                pred = self.model(input)
                pred = pred.unsqueeze(0) # (1, S)

                loss = self.criterion(true, pred)

                if true.shape[0] == self.batch_size:
                    weighted_loss = loss.detach().cpu().numpy()
                else:
                    weighted_loss = loss.detach().cpu().numpy() * self.batch_size / true.shape[0]

                test_loss.append(weighted_loss)

                trues[idx*self.batch_size:(idx+1)*self.batch_size,:] = true.detach().cpu().numpy()
                preds[idx*self.batch_size:(idx+1)*self.batch_size,:] = pred.detach().cpu().numpy()

            test_loss = sum(test_loss) / len(test_loss)
            test_metrics = eval_all_metrics(trues, preds)
            test_acc = test_metrics["acc"]

            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch":self.epoch,
                        f"Test Loss (Phase {self.phase})":test_loss,
                        f"Test ACC (Phase {self.phase})":test_acc,
                    }
                )

        return test_loss, test_metrics, preds
    

    def fit(self):

        if wandb.run is not None:
            print("Generate W&B Artifact!")
            artifact = wandb.Artifact(self.run_name, type='model', metadata=self.config)
            artifact.add_file('./main.py')
            artifact.add_file('./snapshot.py')
            artifact.add_file('./backtest.py')
            artifact.add_file('./data_load.py')
            artifact.add_file('./supervisor.py')
            artifact.add_file('./model.py')
            artifact.add_file('./layer.py')
            artifact.add_dir('./utils')

            for i in range(1,11):
                wandb.define_metric(f"Train Loss (Phase {i})", step_metric="epoch")
                wandb.define_metric(f"Train ACC (Phase {i})", step_metric="epoch")
                wandb.define_metric(f"Test Loss (Phase {i})", step_metric="epoch")
                wandb.define_metric(f"Test ACC (Phase {i})", step_metric="epoch")

        all_phase_results = []

        self.phase = 1
        flag = True

        while flag:

            if add_month_to_string_time(self.end_backtest, self.backtest_month) > self.end_backtest_final:
                flag = False

            print(f"Start Training (Phase {self.phase})...")

            self.train()
            test_performance = self.test()

            res = pd.DataFrame([dict(
                # mae = test_performance[1]["mae"],
                # rmse = test_performance[1]["rmse"],
                ic = test_performance[1]["ic"],
                icir = test_performance[1]["icir"],
                rank_ic = test_performance[1]["rankic"],
                rank_icir = test_performance[1]["rankicir"],
                acc = test_performance[1]["acc"],
                prec = test_performance[1]["long_k_prec"],
                final_return = test_performance[0]["Return"],
                sr = test_performance[0]["SR"],
                mdd = test_performance[0]["MDD"],
            )], index=[f"Phase_{self.phase}"])

            all_phase_results.append(res)

            self.start_train = add_month_to_string_time(self.start_train, self.backtest_month)
            self.end_train = add_month_to_string_time(self.end_train, self.backtest_month)
            self.start_test = add_month_to_string_time(self.start_test, self.backtest_month)
            self.end_test = add_month_to_string_time(self.end_test, self.backtest_month)
            self.start_backtest = add_month_to_string_time(self.start_backtest, self.backtest_month)
            self.end_backtest = add_month_to_string_time(self.end_backtest, self.backtest_month)

            self.phase += 1

        combined_results = pd.concat(all_phase_results)
        mean_values = combined_results.mean(axis=0).to_frame().T
        mean_values.index = ['Mean']
        final_results = pd.concat([combined_results, mean_values])
        final_results = final_results.T
        final_results.index = ['IC', 'ICIR', 'RankIC', 'RankICIR', 'ACC', 'Prec@K', 'Return', 'SR', 'MDD']
        final_results.to_csv(f"{self.res_path}{self.run_name}.csv", index=True)
        
        if wandb.run is not None:
            artifact.add_dir(f"{self.save_path}")
            wandb.log_artifact(artifact)
            print("Upload W&B Artifact!")

            final_results_with_index = final_results.reset_index() # index upload
            res_table = wandb.Table(dataframe=final_results_with_index)
            wandb.log({"Test Result": res_table})
            print("Upload Final Results!")

