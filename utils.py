from pathlib import Path
import torch
import xarray as xr

def resize(tensor, batch=None):
    if len(tensor.shape) == 5 and batch != None:
        return tensor.permute([0,3,4,1,2]).contiguous().view(batch,-1,12,13).permute([0, 2, 3, 1])
    elif len(tensor.shape) == 4 and batch == None:
        return tensor.permute([2,3,0,1]).contiguous().view(-1,12,13).permute([1, 2, 0])

def load_nc_data(subset="train", count=16):
    assert subset in ['train', 'test'], "Subset must be either train or test"

    folder = "/cmlscratch/izvonkov/forecaster-data-processed-split/" + subset
    
    paths = [str(path) for path in list(Path(folder).glob("*.nc")) if str(path).endswith(".nc")]

    if count != -1:
        paths = paths[:count]

    return [torch.tensor(xr.open_dataarray(path).values) for path in paths]

def stack_batch(data, batch_size=16):
    return torch.stack([*data], dim=0)
    

    
