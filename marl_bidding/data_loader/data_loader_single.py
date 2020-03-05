from marl_bidding.data_loader.data_loader_base import DataLoaderBase
import sys
# data_path: load features
# om: With or Without market model
# random_om: With random market distribution / With estimated market distribution
#
# mode = 1: including ipinyou market price
# mode = 2: without ipinyou market price
#

class SingleDataLoader(DataLoaderBase):
    def __init__(self, data_path, camp='2997', file='train.ctr.txt', mode=2):
        self.data_path = '{}{}/{}'.format(data_path, camp, file)
        self.data_source = None
        self.mode = mode
        self.reset()
        # self.temfile_data = open('../log/tmp/data_single.txt', 'w')

    def reset(self):
        self.data_source = open(self.data_path, 'r')

    def get_next(self, mode):
        try:
            raw_bid = next(self.data_source)
            #self.temfile_data.write('1' + '\n')

        except StopIteration:
            #self.temfile_data.write('data done' + '\n')
            self.reset()
            raw_bid = next(self.data_source)
            #self.temfile_data.write('1' + '\n')

        bid = self._construct_bid(raw_bid, mode)
        return bid

    def get_dataset_length(self):
        self.reset()
        length = 0
        for _ in self.data_source:
            length += 1
        self.reset()
        return length

    @staticmethod
    def _construct_bid(raw_bid, mode):
        bid = {}
        raw_bid = raw_bid.strip().split()
        bid['click'] = float(raw_bid[0])
        if mode == 1:
            bid['payprice'] = float(raw_bid[1])
        bid['pctr'] = float(raw_bid[2])
        try:
            bid['bid'] = list(map(int, raw_bid[3:23]))
        except ValueError:
            # for the test.ctr__ file
            convert_float = list(map(float, raw_bid[3:23]))
            bid['bid'] = list(map(int, convert_float))
        if len(raw_bid) > 23:
            bid['market_price'] = list(map(float, raw_bid[23:]))
        return bid

    def close(self):
        self.data_source.close()