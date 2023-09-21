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


DEFAULT_SIZE_FN = lambda x: x.size(0)

def cliped_iter_dataloder(dataloader, num_samples: int = None, size_fn: callable = DEFAULT_SIZE_FN):
    """
    Clips the dataloader to num_samples
    """
    if num_samples is None:
        yield from dataloader
        return
    
    remaining_samples = num_samples
    for batch in dataloader:
        x, *rem = batch
        if remaining_samples < size_fn(x):
            if isinstance(x, torch.Tensor):
                x = x[:remaining_samples]
            elif isinstance(x, dict):
                x = {k: v[:remaining_samples] for k, v in x.items()}
            batch = x, *[rr[:remaining_samples] if rr is not None else None for rr in rem ]
            yield batch
            return
        remaining_samples -= size_fn(x)
        yield batch



def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    else:
        raise NotImplementedError(f"Unknown type: {type(x)}")