# 单机多GPU

主要实践在: 单机多卡。

分布式训练的思想是: 基于多进程，将数据拆分到不同的进程，进行训练。Pytorch 提供了在不同的进程中
对参数进行合并的操作。

## 训练启动器 - Launcher

`Launcher` 用来启动训练过程。需要子类实现自己的 `Luancher`。特别需要注意的是: 因为是多进程启动训练，
那么，会导致 `Luancher` 中的成员变量复制到其他进程中, 而这个过程会由于某些类无法复制到其他进程而引发
一些非常难以解决的问题。所以，对于子类中的 `Luancher` 尽量不要使用任何成员变量。


一些组件，必须遵循分布式运算规范。如下:

| 组件名称  | 是否需要继承 `easytext.distributed.Distributed` 类 | 备注 |
| ---------|-------------------|
| DataLoader | 否 |
| Model | 否 |
| Loss | 否 |
| Metric | 是 |
| `easytext.trainer.TrainerCallback` | 是 |

关于 `easytext.distributed.Distributed` 说明, 在多GPU训练中会检查组件是否继承该类。
其中 `easytext.distributed.Distributed.is_distributed` 的含义是: 当前训练是否在
多GPU训练状态下，相应的组件要根据该状态进行编写相应的代码逻辑。
如果上表中 "是否需要继承 `easytext.distributed.Distributed` 类" 为 "是",
没有继承 `Distributed`, 那么，则不能用在多GPU的训练中使用; 而继承了 `Distributed`,
则可以在 单 GPU/CPU 和 多 GPU 中使用。


基于上面的思想，在写分布式训练的时候，需要解决的一些组件包括:

1. `DataLoader` 如何运行?
2. `Model` 如何参数更新?
3. `Loss` 如何计算?
4. `Metric` 如何计算?

##  `DataLoader` 如何运行?

使用 `torch.utils.data.distributed.DistributedSampler` 根据当前进程对 dataset 进行采样。
类似下面这段代码:

```python

import torch

def create_dataloader(distributed: bool, dataset: torch.data.Dataset, batch_size: int):
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    return data_loader
```

关于 `batch_size` 表示每一个 GPU 处理的 `batch_size`, 如果有 N 个 GPU, 那么, 就能同时处理 `batch_size * N`.

##  `Model` 如何参数更新?
Model, 训练矿建会自动使用 `torch.nn.parallel.DistributedDataParallel(model)` 设置，所以实际代码按照单GPU写即可。

##  `Loss` 如何计算?
Loss 会在 Trainer 中自动进行 分布式计算，所以正常写即可。

##  `Metric` 如何计算?

Metric 由于会对多 GPU 运算的结果进行汇总，所以需要继承  `Synchronized`,
实现 `to_synchronized_data` 和 `from_synchronized_data` 来设置需要同步的数据。

## 训练需要其他参数

由于使用了 pytorch 的分布式训练作为单机多GPU训练方案，那么，就需要对 `init_process_group` 
参数提供一个 `init_method` 参数, 该参数默认使用了 TCP 协议，所以，需要提供一个空闲的 PORT，
这是需要在训练的时候指定的。并没有使用 ENV 协议。

另外，说明一下关于使用设置本机环境变量。

1. 设置 `CUDA_VISIBLE_DEVICES`, 可以设置，如果设置了，要注意在配置训练的 GPU 时候，特别注意序号. 如果不配置就会正常使用训练时候配置的 gpu 参数。
2. 由于在 `init_process_group` 使用了 TCP 协议，所以不需要设置任何进程组相关环境变量，设置了也不会生效。

## 单机CPU多进程训练
无法进行单机CPU多进程训练。

## 相关工具

### `DistributedFuncWrapper`

`from easytext.utils.distributed_util import DistributedFuncWrapper`

该类能够让某个函数，只在指定的进程中执行，这是很有用处的。比如，写文件或者日志输出，
只需要在一个进程中输出。如果指定 `dst_rank=None`, 那么，则不会进行进程处理，这样，
同样的代码，可以在单 GPU/CPU 下也可以运行了。



