import torch
# context manager for setting the random seed
class DeterministicRandomness:
    def __init__(self, seed):
        self.seed = seed
        self.rng_state = None

    def __enter__(self):
        self.rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_rng_state(self.rng_state)



def cliped_iter_dataloder(dataloader, num_samples: int = None):
    """
    Clips the dataloader to num_samples
    """
    if num_samples is None:
        yield from dataloader
        return
    
    remaining_samples = num_samples
    for batch in dataloader:
        x, *rem = batch
        if remaining_samples < len(x):
            batch = x[:remaining_samples], *[rr[:remaining_samples] for rr in rem]
            yield batch
            return
        remaining_samples -= len(x)
        yield batch