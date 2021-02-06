# deep-sets
This repository contains an experimental re-implementation of Deep Sets architecture, proposed in 2017 paper [Deep Sets](https://arxiv.org/pdf/1703.06114.pdf). 

## Deep Sets
Deep sets architecture aims to approximate functions acting on a set. To be more mathematically accurate, deep sets will approximate a mapping from a power set of some set to a target space (real space for regression and discrete space for classification task). The most notable contribution of such architectures are:

- Can learn permutation invariant / equivariant functions
- Can take inputs of different length (must be padded accordingly)
- Can perform significantly better in certain set-related tasks at lower computation costs

### Permutation Invariant Functions
A permutation invariant function acting on a set input is defined to be a function that does not change its output due to permutation on the input elements.
<center>
<img src="https://github.com/jkgrad/deep-sets/blob/main/assets/perm-invariant.png" height="25">
</center>

Such functions can be decomposed in the following form, which we can exploit in building the neural network architecture. 
<center>
  <img src="https://github.com/jkgrad/deep-sets/blob/main/assets/perm-invariant-decompose.png" height="60">
</center>

### Permutation Equivariant Functions
A permutation equivariant function acting on a set input is defined to be a function that permutes its output in the same order as the permutation on its inputs.
<center>
  <img src="https://github.com/jkgrad/deep-sets/blob/main/assets/perm-equivariant.png" height="25">
</center>

## Notes
The current implementation includes a permutation invariant function approximator on set of digits with some more rooms to improve. Note that following architecture was experimented for approximating a sum of digits. 

```python
DigitFuncApproximator(
  (embed): Embedding(11, 100, padding_idx=0)                   # Phi
  (dense): Linear(in_features=100, out_features=30, bias=True) # Phi
  (actvt): Tanh()                                              # Phi
  (lmbda): lambda x: torch.sum(x, dim=1)                       # Aggregator
  (encde): Linear(in_features=30, out_features=1, bias=True)   # Rho
)
```

