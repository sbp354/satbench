import torch.nn as nn
import torch
import numpy as np

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # print(true_dist)
        # return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return true_dist

p = torch.Tensor([[1.2,3.4,5.2,2,3,4,5,6,7,8]])
y = torch.tensor([6])
yh = torch.tensor([[0,1,0], [0,0,1]])

# print(nn.CrossEntropyLoss()(p, y))
import matplotlib.pyplot as plt
for s in np.arange(0.0, 0.91, 0.9/6):
    print(s)
    true_dist =np.array(LabelSmoothingLoss(10, smoothing=s)(p, y)[0])
    print(true_dist)
    plt.bar(list(range(0,10)), height=true_dist)
    plt.yticks(np.arange(0.0,1.1,0.1))
    plt.show()