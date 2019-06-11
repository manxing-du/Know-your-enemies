class DataLoaderBase(object):
    def reset(self):
        raise NotImplementedError

    def get_next(self, mode):
        raise NotImplementedError

    def get_batch(self):
        raise NotImplementedError

    def get_dataset_length(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError