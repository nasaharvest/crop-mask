from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from datetime import datetime
import xarray as xr


from src.engineer.base import BaseEngineer


def sentinel_as_tci(sentinel_ds: xr.DataArray, scale: bool = True) -> xr.DataArray:
    r"""
    Get a True Colour Image from Sentinel data exported from Earth Engine
    :param sentinel_ds: The sentinel data, exported from Earth Engine
    :param scale: Whether or not to add the factor 10,000 scale
    :return: A dataframe with true colour bands
    """

    band2idx = {band: idx for idx, band in enumerate(sentinel_ds.attrs["band_descriptions"])}

    tci_bands = ["B4", "B3", "B2"]
    tci_indices = [band2idx[band] for band in tci_bands]
    if scale:
        return sentinel_ds.isel(band=tci_indices) / 10000 * 2.5
    else:
        return sentinel_ds.isel(band=tci_indices) * 2.5


def plot_results(model_preds: xr.Dataset, tci_path: Path, savepath: Path, prefix: str = "") -> None:

    multi_output = len(model_preds.data_vars) > 1

    tci = sentinel_as_tci(
        BaseEngineer.load_tif(tci_path, start_date=datetime(2020, 1, 1), days_per_timestep=30),
        scale=False,
    ).isel(time=-1)

    tci = tci.sortby("x").sortby("y")
    model_preds = model_preds.sortby("lat").sortby("lon")

    plt.clf()
    fig, ax = plt.subplots(1, 3, figsize=(20, 7.5), subplot_kw={"projection": ccrs.PlateCarree()})

    fig.suptitle(
        f"Model results for tile with bottom left corner:"
        f"\nat latitude {float(model_preds.lat.min())}"
        f"\n and longitude {float(model_preds.lon.min())}",
        fontsize=15,
    )
    # ax 1 - original
    img_extent_1 = (tci.x.min(), tci.x.max(), tci.y.min(), tci.y.max())
    img = np.clip(np.moveaxis(tci.values, 0, -1), 0, 1)

    ax[0].set_title("True colour image")
    ax[0].imshow(img, origin="upper", extent=img_extent_1, transform=ccrs.PlateCarree())

    args_dict = {
        "origin": "upper",
        "extent": img_extent_1,
        "transform": ccrs.PlateCarree(),
    }

    if multi_output:
        mask = np.argmax(model_preds.to_array().values, axis=0)

        # currently, we have 10 classes (at most). It seems unlikely we will go
        # above 20
        args_dict["cmap"] = plt.cm.get_cmap("tab20", len(model_preds.data_vars))
    else:
        mask = model_preds.prediction_0
        args_dict.update({"vmin": 0, "vmax": 1})

    # ax 2 - mask
    ax[1].set_title("Mask")
    im = ax[1].imshow(mask, **args_dict)

    # finally, all together
    ax[2].set_title("Mask on top of the true colour image")
    ax[2].imshow(img, origin="upper", extent=img_extent_1, transform=ccrs.PlateCarree())

    args_dict["alpha"] = 0.3
    if not multi_output:
        mask = mask > 0.5
    ax[2].imshow(mask, **args_dict)

    colorbar_args = {
        "ax": ax.ravel().tolist(),
    }

    if multi_output:
        # This function formatter will replace integers with target names
        formatter = plt.FuncFormatter(lambda val, loc: list(model_preds.data_vars)[val])
        colorbar_args.update({"ticks": range(len(model_preds.data_vars)), "format": formatter})

    # We must be sure to specify the ticks matching our target names
    fig.colorbar(im, **colorbar_args)

    plt.savefig(savepath / f"results_{prefix}{tci_path.name}.png", bbox_inches="tight", dpi=300)
    plt.close()
