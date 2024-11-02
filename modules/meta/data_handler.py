from typing import Any, Optional, Callable, Iterator, Sized

import numpy as np

from torch.utils.data import Sampler, Dataset, DataLoader


class CyclicSampler(Sampler):
    def __init__(
        self,
        dataset_size: int,
        order: Optional[Sized] = None,
        loops: int = 1,
    ):
        super().__init__()
        assert loops > 0
        assert order is None or len(order) == dataset_size
        self.dataset_size = dataset_size
        self.order = order if order is not None else range(dataset_size)
        self.loops = loops

    def _iterator(self):
        for _ in range(self.loops):
            for j in self.order:
                yield j

    def __iter__(self) -> Iterator[int]:
        return iter(self._iterator())

    def __len__(self):
        return self.dataset_size * self.loops


class DataHandler:
    def __init__(
        self,
        data_source: Dataset,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        self._iter = Iterator[Any]
        self._collate_fn = collate_fn
        self._batch_size = batch_size
        self._loops = loops
        self._init_impl(data_source, seed, batch_size, loops, collate_fn)

    def _init_impl(
        self,
        data_source: Dataset,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        rng = np.random.RandomState(seed)
        dataset_size = len(data_source)  # noqa todo
        order = rng.permutation(dataset_size)
        sampler = CyclicSampler(dataset_size, order, loops=loops)
        if collate_fn:
            self._data_loader = DataLoader(
                data_source,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            self._data_loader = DataLoader(
                data_source,
                batch_size=batch_size,
                sampler=sampler,
            )
        self._iter = iter(self._data_loader)

    def __iter__(self):
        return self._iter

    def __next__(self):
        # print(next(self._iter))
        return next(self._iter)

    def reset(self):
        self._iter = iter(self._data_loader)

    def seed(self, seed: int):
        self._init_impl(
            self._data_loader.dataset,
            seed,
            self._batch_size,
            self._loops,
            self._collate_fn,
        )


# def test_random_cyclic_sampler_default_order():
#     alist = [0, 1, 2]
#     sampler = CyclicSampler(len(alist), None, loops=10)
#     cnt = 0
#     for i, x in enumerate(sampler):
#         assert alist[x] == i % 3
#         cnt += 1
#     assert cnt == 30
#
#
# def test_random_cyclic_sampler_default_given_order():
#     alist = [1, 2, 0]
#     sampler = CyclicSampler(len(alist), order=[2, 0, 1], loops=10)
#     cnt = 0
#     for i, x in enumerate(sampler):
#         assert alist[x] == i % 3
#         cnt += 1
#     assert cnt == 30
#
#
# def test_data_handler():
#     class SimpleDataset(Dataset):
#         def __init__(self):
#             self.data = list(range(10))
#
#         def __len__(self):
#             return len(self.data)
#
#         def __getitem__(self, idx):
#             return self.data[idx]
#
#     data = SimpleDataset()
#     batch_size = 2
#     loops = 3
#     handler = DataHandler(data, None, batch_size=batch_size, loops=loops)
#     cnt = dict([(x, 0) for x in data])
#     for x in handler:
#         print(f"Batch: {x}, Length: {len(x)}")
#         assert len(x) == batch_size
#         for t in x:
#             v = t.item()
#             cnt[v] = cnt[v] + 1
#     for x in cnt:
#         assert cnt[x] == loops
