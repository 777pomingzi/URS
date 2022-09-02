from .popular import PopularNegativeSampler
from .random import RandomNegativeSampler


NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, test, user_count, item_count, sample_size, seed, save_folder,users,items):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(test, user_count, item_count, sample_size, seed, save_folder,users,items)
