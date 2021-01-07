from torch.utils.data import DataLoader


class Client:
    def __init__(self, cid, train_data, test_data, bs, worker):
        self.cid = cid
        self.worker = worker

        self.train_data = train_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)

    def get_model_params(self):
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def train_client(self, **kwargs):
        local_solution, worker_stats = self.worker.train(self.train_dataloader, **kwargs)
        return local_solution, worker_stats

    def test_client(self, use_eval_data=True):
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        tot_correct, loss = self.worker.test(dataloader)
        return tot_correct, len(dataset), loss
