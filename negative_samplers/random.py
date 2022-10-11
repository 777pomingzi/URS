from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    # def generate_negative_samples(self):#生成random-mask的负样本
    #     #print(self.seed)
    #     assert self.seed is not None, 'Specify seed for random sampling'
    #     np.random.seed(self.seed)
    #     negative_samples = {}
    #     print('Sampling negative items')
    #     for idx in trange(0, self.user_count):
    #         user=self.users[idx]
    #         if isinstance(self.test[user][1], tuple):
    #             seen = set(x[0] for x in self.test[user])
    #             #seen = set(x[0] for x in self.train[user])
    #             #seen.update(x[0] for x in self.val[user])
    #             #seen.update(x[0] for x in self.test[user])
    #         else:
    #             seen = set(self.test[user])
    #             #seen = set(self.train[user])
    #             #seen.update(self.val[user])
    #             #seen.update(self.test[user])

    #         samples = []
    #         for _ in range(self.sample_size):
    #             item = self.items[np.random.choice(self.item_count)]
    #             prefix_item='<extra_id_0> '+item
    #             while item in seen or prefix_item in samples:
    #                 item = self.items[np.random.choice(self.item_count)]
    #                 prefix_item='<extra_id_0> '+item
    #             samples.append(prefix_item)

    #         negative_samples[user] = samples

    #     return negative_samples


    def generate_negative_samples(self):#leave one out的负样本
        #print(self.seed)
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for idx in trange(0, self.user_count):
            user=self.users[idx]
            if isinstance(self.test[user][1], tuple):
                seen = set(x[0] for x in self.test[user][1:])
                #seen = set(x[0] for x in self.train[user])
                #seen.update(x[0] for x in self.val[user])
                #seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.test[user][1:])
                #seen = set(self.train[user])
                #seen.update(self.val[user])
                #seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = self.items[np.random.choice(self.item_count)]
                while item in seen or item in samples:
                    item = self.items[np.random.choice(self.item_count)]
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples
