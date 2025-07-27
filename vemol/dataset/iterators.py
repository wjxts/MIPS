import itertools
import math 


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by
            ``__len__``. This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self.iterable = iterable
        self.itr = iter(self)

        if start is None:
            self.n = getattr(iterable, 'n', 0)
        else:
            self.n = start

        if total is None:
            self.total = self.n + len(iterable)
        else:
            self.total = total

    def __len__(self):
        return self.total

    def __iter__(self):
        for x in self.iterable:
            if self.n >= self.total:
                raise RuntimeError(
                    'Mismatch between actual and expected iterable length. '
                    'Please report this to the fairseq developers.'
                )
            self.n += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        if hasattr(self.iterable, "take"):
            self.iterable.take(n)
        else:
            self.iterable = itertools.islice(self.iterable, n)

class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size):
        itr = _chunk_iterator(iterable, chunk_size)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, 'n', 0) / float(chunk_size))),
            total=int(math.ceil(len(iterable) / float(chunk_size))),
        )
        self.chunk_size = chunk_size


def _chunk_iterator(itr, chunk_size):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk