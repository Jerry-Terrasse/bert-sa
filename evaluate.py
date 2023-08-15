import torch
from matplotlib import pyplot as plt

def discriminate(X: torch.Tensor):
    return torch.round(X * 10).long()

class Result:
    def __init__(self, label: torch.Tensor, pred: torch.Tensor):
        self.label = label
        self.pred = pred
        self.label_level = discriminate(label)
        self.pred_level = discriminate(pred)
    def summary(self, tolerance: int = 0):
        result = ((self.label_level - self.pred_level).abs() <= tolerance).long()
        accuracy = result.float().mean()
        return accuracy, result
    def table(self):
        # result = torch.zeros(11, 11, dtype=torch.long).to(self.label.device)
        # result[self.label_level, self.pred_level] += 1
        n = 11
        result = torch.zeros(n, n, dtype=torch.long).to(self.label.device)
        indices = self.label_level * n + self.pred_level
        counts = torch.bincount(indices)
        result.view(-1)[: counts.numel()] = counts
        return result
    def __add__(self, other: 'Result'):
        return Result(
            torch.cat([self.label, other.label]),
            torch.cat([self.pred, other.pred])
        )
    def __iadd__(self, other: 'Result'):
        self.label = torch.cat([self.label, other.label])
        self.pred = torch.cat([self.pred, other.pred])
        self.label_level = torch.cat([self.label_level, other.label_level])
        self.pred_level = torch.cat([self.pred_level, other.pred_level])
        return self
    @staticmethod
    def heatmap(table_: torch.Tensor, save_path: str):
        assert table_.shape == (11, 11)
        plt.figure(figsize=(10, 10))
        plt.xlabel('Predicted Level')
        plt.ylabel('Label Level')
        plt.title('Confusion Matrix')
        plt.imshow(table_.cpu())
        plt.colorbar()
        plt.savefig(save_path)
    @staticmethod
    def reduce(results: list['Result']):
        res = Result.empty(results[0].label.device)
        for result in results:
            res += result
        return res
    @staticmethod
    def empty(device: torch.device):
        return Result(
            torch.Tensor([]).to(device),
            torch.Tensor([]).to(device)
        )