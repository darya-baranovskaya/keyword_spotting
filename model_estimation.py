import time
from thop import profile 
import wandb
import torch

def get_size_in_megabytes(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size = next(model.parameters()).element_size()
    return (num_params * param_size) / (2 ** 20)


class Timer:

    def __init__(self, name: str, verbose=False):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"{self.name.capitalize()} | Elapsed time : {self.t:.10f}")
            

def estimate_time(estimation_model):
    tt = torch.rand((256, 40, 80))
    with torch.no_grad():
        with Timer("main model", verbose=False) as t_test:
            gg = estimation_model(tt)
        with Timer("main model", verbose=False) as t:
            gg = estimation_model(tt)
        with torch.no_grad():
            with Timer("main model 10 runs", verbose=True) as t10:
                for i in range(10):
                    gg = estimation_model(tt)
    return t.t, t10.t
            
def estimate_model_complexity(model):
    macs, params = profile(model, (torch.rand(1, 40, 101), ))
    macs_batch, params_batch = profile(model, (torch.rand(128, 40, 101), ))
    t, t10 = estimate_time(model)
    size_in_megabytes = get_size_in_megabytes(model)
    result_dict = {'time_on_one_frame': t, 'time_on_10_frames': t10, 'macs': macs,
               'params':params, 'size in megabytes':size_in_megabytes,
                  'macs_batch': macs_batch, 'params_batch': params_batch}
    wandb.log(result_dict)
    return result_dict
    