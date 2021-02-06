import torch
import torch.nn as nn

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