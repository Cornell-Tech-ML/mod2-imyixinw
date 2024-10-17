[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py


|        | Simple |  Diag  | Split | Xor |
|:-------|:------:|:------:|:-----:|----:|
| # of points | 50 | 50 | 50 | 50 |
| size of hidden layer | 4 | 6 | 6 | 9 |
| learning rate | 0.1 | 0.1 | 0.1 | 0.5 |
| # of epochs | 500 | 500 | 575 | 500 |
| time per epoch | 0.066s | 0.110s | 0.110s | 0.199s

<img src="fig/1simple.png">
<img src='fig/2diag.png'>
<img src='fig/3split.png'>
<img src='fig/4xor.png'>