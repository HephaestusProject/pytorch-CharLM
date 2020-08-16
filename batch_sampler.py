from torch.utils.data import BatchSampler, Sampler


class SequentialBatchSampler(BatchSampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super(SequentialBatchSampler, self).__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )

        assert drop_last == True, "drop_last must be True for SequentialBatchSampler"

    def __iter__(self):
        num_batches = len(self)
        batch = []
        for i in range(num_batches):
            for b in range(self.batch_size):
                sequence_head_position = num_batches * b
                batch.append(sequence_head_position + i)
            yield batch
            batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
