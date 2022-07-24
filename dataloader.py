from multiprocessing import Process, Queue
from queue import Empty
import random
import time


def default_collate_fn(elements):
    return elements


def worker_fn(dataset, index_queue, data_queue, collate_fn):
    while True:
        try:
            index_info = index_queue.get(block=True, timeout=10.)
        except Empty:
            continue
        else:
            if index_info is None:
                return

        iteration_count, batch_index, indices = index_info
        data = collate_fn([dataset[index] for index in indices])
        data_queue.put((iteration_count, batch_index, data))


class DataLoader:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 collate_fn=None,
                 num_workers=1,
                 shuffle=False,
                 drop_last=True,
                 prefetch_factor=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.num_batches += int(not drop_last and
                                (len(dataset) % batch_size > 0))
        self.collate_fn = collate_fn or default_collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.workers = []

    def _maybe_put_indices(self):
        target_inflight_batch_count = self.num_workers * self.prefetch_factor
        while (self.num_sent_indices < len(self.indices) and
               self.inflight_batch_count < target_inflight_batch_count):
            num_elements_in_batch = min(
                len(self.indices) - self.num_sent_indices,
                self.batch_size
            )
            indices = self.indices[
                self.num_sent_indices:
                self.num_sent_indices + num_elements_in_batch
            ]

            self.index_queues[self.next_worker].put(
                (self.iteration_count, self.sent_batch_index, indices))
            self.num_sent_indices += num_elements_in_batch
            self.sent_batch_index += 1
            self.inflight_batch_count += 1
            self.next_worker = (self.next_worker + 1) % self.num_workers

    def _maybe_start(self):
        if self.workers:
            return

        self.data_queue = Queue()
        self.index_queues = []
        self.workers = []
        for _ in range(self.num_workers):
            index_queue = Queue()
            worker = Process(
                target=worker_fn,
                args=(self.dataset, index_queue, self.data_queue,
                      self.collate_fn),
                daemon=True
            )
            worker.start()
            self.index_queues.append(index_queue)
            self.workers.append(worker)

    def _reset(self):
        self.num_sent_indices = 0
        self.sent_batch_index = 0
        self.current_batch_index = 0
        self.inflight_batch_count = 0
        self.iteration_count = getattr(self, 'iteration_count', -1) + 1
        self.cache = {}
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        if self.drop_last:
            self.indices = self.indices[:self.num_batches * self.batch_size]
        self.next_worker = 0
        self._maybe_put_indices()

    def __iter__(self):
        self._maybe_start()
        self._reset()
        return self

    def __next__(self):
        if self.current_batch_index >= self.num_batches:
            raise StopIteration

        self._maybe_put_indices()

        while True:
            if self.current_batch_index in self.cache:
                data = self.cache.pop(self.current_batch_index)
                self.current_batch_index += 1
                return data

            try:
                iteration_count, batch_index, data = \
                    self.data_queue.get(block=False, timeout=0.5)
            except Empty:
                continue
            else:
                if iteration_count == self.iteration_count:
                    self.cache[batch_index] = data
                    self.inflight_batch_count -= 1

    def __del__(self):
        if self.workers:
            for index_queue in self.index_queues:
                index_queue.put(None)
            for worker in self.workers:
                worker.join()
            for index_queue in self.index_queues:
                index_queue.cancel_join_thread()
                index_queue.close()
            self.data_queue.cancel_join_thread()
            self.data_queue.close()


def test():
    ds = list(range(59))
    dl = DataLoader(ds, batch_size=3, num_workers=2,
                    prefetch_factor=2, shuffle=True, drop_last=False)
    retrieved = []
    for batch in dl:
        retrieved.extend(batch)
    assert sorted(ds) == sorted(retrieved)


def benchmark():
    ds = list(range(105904))
    dl = DataLoader(ds, batch_size=8, num_workers=2,
                    prefetch_factor=2, shuffle=True, drop_last=False)
    start_time = time.perf_counter()
    for _ in dl:
        pass
    end_time = time.perf_counter()
    print(f'{end_time - start_time:3.2f}s')


if __name__ == '__main__':
    test()
    benchmark()
