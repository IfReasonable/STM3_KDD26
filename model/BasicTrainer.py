import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm
class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.device = args.device
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        batch_num = len(val_dataloader)
        pbar = tqdm(enumerate(val_dataloader), total=batch_num)

        with torch.no_grad():
            for batch_idx, (data, target) in pbar:
                data = data.to(self.device)
                label = target[..., :self.args.output_dim].to(self.device)
                output = self.model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                pbar.update(1)

        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(
            epoch, val_loss, time.time() - epoch_time))
        return val_loss

    def test_epoch(self, epoch, test_dataloader):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        batch_num = len(test_dataloader)
        pbar = tqdm(enumerate(test_dataloader), total=batch_num)

        mae_sum, rmse_sum, mape_sum, mse_sum, total_count = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch_idx, (data, target) in pbar:
                data = data.to(self.device)
                label = target[..., :self.args.output_dim].to(self.device)
                output = self.model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
                batch_mae, batch_rmse, batch_mape, batch_mse = All_Metrics(
                    output.detach(), label.detach(),
                    self.args.mae_thresh, self.args.mape_thresh
                )
                batch_count = label.numel()
                mae_sum += batch_mae * batch_count
                rmse_sum += batch_rmse * batch_count
                mape_sum += batch_mape * batch_count
                mse_sum += batch_mse * batch_count
                total_count += batch_count
                pbar.update(1)

        mae = mae_sum / total_count
        rmse = rmse_sum / total_count
        mape = mape_sum / total_count
        mse = mse_sum / total_count
        self.logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, MSE: {:.4f}".format(
                    mae, rmse, mape, mse))
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss, time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        
        batch_num = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader), total=batch_num)
        
        for batch_idx, (data, target) in pbar:
            data = data
            label = target[..., :self.args.output_dim]
            
            data = data.to(self.device)
            label = label.to(self.device)
            
            self.optimizer.zero_grad()

            output = self.model(data)
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)
                label = self.scaler.inverse_transform(label)

            input_data = data[..., :self.args.input_dim]
            loss = self.loss(output, label)
            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx+1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx+1, self.train_per_epoch, loss.item()))
            
            pbar.update(1)

        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, train time: {:.2f} s'.format(epoch, train_epoch_loss,
                                                                                         time.time() - epoch_time))

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_test_model =None
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        vaild_loss = []
        test_loss = []
        train_time = []
        train_M = []

        for epoch in range(0, self.args.epochs):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            vaild_loss.append(val_epoch_loss)

            test_epoch_loss = self.test_epoch(epoch, test_dataloader)
            
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

            if test_epoch_loss< best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())


        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        T = args.horizon
        mae_sum = np.zeros(T)
        rmse_sum = np.zeros(T)
        mape_sum = np.zeros(T)
        mse_sum = np.zeros(T)
        sample_num = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(args.device)
                label = target[..., :args.output_dim].to(args.device)
                output = model(data)
                if args.real_value:
                    output = scaler.inverse_transform(output)
                    label = scaler.inverse_transform(label)
                for t in range(T):
                    batch_mae, batch_rmse, batch_mape, batch_mse = All_Metrics(
                        output[:, t].detach(), label[:, t].detach(),
                        args.mae_thresh, args.mape_thresh
                    )
                    batch_count = label.numel()
                    mae_sum[t] += batch_mae * batch_count
                    rmse_sum[t] += batch_rmse * batch_count
                    mape_sum[t] += batch_mape * batch_count
                    mse_sum[t] += batch_mse * batch_count
                sample_num += batch_count
        mae = mae_sum / sample_num
        rmse = rmse_sum / sample_num
        mape = mape_sum / sample_num
        mse = mse_sum / sample_num
        for t in range(T):
            logger.info(f"Horizon {t}: MAE: {mae[t]:.4f}, RMSE: {rmse[t]:.4f}, MAPE: {mape[t]:.4f}, MSE: {mse[t]:.4f}")

        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, MSE: {:.4f}".format(
            mae.mean(), rmse.mean(), mape.mean(), mse.mean()))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))