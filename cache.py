import torch


def init(use_posthoc):
    global cached_data
    global variables

    cached_data = {}
    variables = {
        "should_use_cache": False,
        "enable_caching": use_posthoc,
        "num_cache_hits": torch.tensor(0.001),
        "num_cache_access": torch.tensor(0),
    }
