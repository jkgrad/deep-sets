import torch
import torch.nn as nn

from typing import Callable, Dict

from tqdm import tqdm
from torch.utils.data import Dataset

class DigitFuncDataset(Dataset):
    """
        Digit function dataset for Deep Sets architecture.
        An example consists of at most `max_n_elements` #
        of digits. If there are less than `max_n_elements`
        in a given example, then the rest is padded. The
        labels are the outputs of the function acting on a
        given sample.

        NOTE:
            A sample would be in the following format:
            - INPUT: {1, 2, 3} -> LABEL: function({1, 2, 3})
            - INPUT: {3, 4, -} -> LABEL: function({3, 4, -})

        Parameters
        -----------
            - n_samples: `int`
                - number of samples.
            - max_n_elements: `int`
                - maximum number of elements in a given sample.
            - function: `Callable`
    """
    def __init__(self, n_samples: int, max_n_elements: int, function: Callable):
        super().__init__()

        self.n_samples = n_samples
        self.inputs = torch.zeros([n_samples, max_n_elements])
        self.labels = torch.zeros([n_samples])
        for i in tqdm(range(n_samples), desc='Generating {} samples ...'.format(n_samples)):
            # # of elements in this given set (multi-set)
            n = torch.randint(low=1, high=max_n_elements, size=[1])
            for j in range(n):
                r = torch.randint(low=1, high=10, size=[1])
                self.inputs[i,j] = r
            self.labels[i], _ = function(self.inputs[i,:], dim=-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        return \
            {
                "input": self.inputs[idx].long(),
                "label": self.labels[idx].long()
            }

class DigitFuncApproximator(nn.Module):
    """
        A Deep Set network for approximating a function
        acting on a set of digits. The architecture 
        follows the permutation invariant model from the
        paper.

        NOTE: A general architecture for permutation 
              invariant networks has the following form
              
            f(\vec{x}) = \rho (\sum_{x \in X} \phi(\vec{x}))
            
                \phi : Embedding -> Linear -> Activation
                \rho : Linear
    """
    def __init__(self, d_embed: int, d_hidden: int, activation: str):
        super().__init__()

        # 0 - padding, 1~10 - inputs 
        self.embed = nn.Embedding(11, d_embed, padding_idx=0)
        self.dense = nn.Linear(d_embed, d_hidden)
        self.actvt = getattr(nn, activation)()
        self.lmbda = lambda x: torch.sum(x, dim=1)
        self.encde = nn.Linear(d_hidden, 1)

    def forward(self, x: "torch.Tensor"):
        x = self.embed(x)
        x = self.dense(x)
        x = self.actvt(x)
        x = self.lmbda(x)
        x = self.encde(x)
        return x