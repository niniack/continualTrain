from contextlib import contextmanager

import torch


@contextmanager
def conditional_profiler(flag, *args, **kwargs):
    if flag:
        with torch.profiler.profile(*args, **kwargs) as prof:
            yield prof
    else:
        yield None


def print_profiler_results(prof, sort_by="cuda_time_total"):
    if prof is not None:
        print(prof.key_averages().table(sort_by=sort_by))
