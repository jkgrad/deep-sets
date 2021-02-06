import os
import pickle
import torch

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
    def __init__(self, n_samples: int, max_n_elements: int, function: Callable, path: str):
        super().__init__()

        self.n_samples = n_samples
        if os.path.isfile(path):
            info = pickle.load(open(path, 'rb'))
            self.inputs = info['inputs']
            self.labels = info['labels']
        else:
            self.inputs = torch.zeros([n_samples, max_n_elements])
            self.labels = torch.zeros([n_samples])
            for i in tqdm(range(n_samples), desc='Generating {} samples ...'.format(n_samples)):
                # # of elements in this given set (multi-set)
                n = torch.randint(low=1, high=max_n_elements, size=[1])
                for j in range(n):
                    r = torch.randint(low=1, high=10, size=[1])
                    self.inputs[i,j] = r
                output = function(self.inputs[i,:])
                if isinstance(output, tuple):
                    self.labels[i], _ = output
                else:
                    self.labels[i] = output
            # pickle.dump({'inputs': self.inputs, 'labels': self.labels}, open(path, 'wb'))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        return \
            {
                "input": self.inputs[idx].long(),
                "label": self.labels[idx].long()
            }