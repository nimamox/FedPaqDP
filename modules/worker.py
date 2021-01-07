import torch.nn as nn
import torch

from utils.torch_utils import get_flat_params_from, set_flat_params_to


criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker:
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.local_iters = args['local_iters']
        self.gpu = args['gpu']

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def train(self, train_dataloader, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        for iter in range(self.local_iters):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_soln = self.get_flat_model_params()
        stat_dict = {"loss": train_loss / train_total,
                     "acc": train_acc / train_total}

        return local_soln, stat_dict

    def test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss
