import abc


# TODO: dataclass? but we want nontrivial constructors in base classes.
class GridExample:
    def __init__(self, name, input, target, metadata):
        self.name = name
        self.input = input
        self.target = target
        self.metadata = metadata


class Dataset:
    @abc.abstractmethod
    def __len__(self):
        pass

    @property
    @abc.abstractmethod
    def names(self):
        pass

    @property
    def nchannels(self):
        return 1

    @abc.abstractmethod
    def __getitem__(self, i):
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class PostprocessDataset(Dataset):
    def __init__(self, base_dataset, postprocess_fn, nchannels):
        self._base_dataset = base_dataset
        self._postprocess = postprocess_fn
        self._nchannels = nchannels

    @property
    def names(self):
        return self._base_dataset.names

    @property
    def nchannels(self):
        return self._nchannels

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, i):
        return self._postprocess(self._base_dataset[i])
