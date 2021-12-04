from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Optional

import os
import re


def gdal_cmd(cmd_type: str, in_file: str, out_file: str, msg: Optional[str] = None):
    if cmd_type == "gdalbuildvrt":
        cmd = f"gdalbuildvrt {out_file} {in_file}"
    elif cmd_type == "gdal_translate":
        cmd = f"gdal_translate -a_srs EPSG:4326 -of GTiff {in_file} {out_file}"
    elif cmd_type == "gdal_translate_cog":
        cmd = (
            f"gdal_translate -a_srs EPSG:4326 -of GTiff {in_file} {out_file} "
            + "-co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co COMPRESS=LZW"
        )
    else:
        raise NotImplementedError(f"{cmd_type} not implemented.")
    if msg:
        print(msg)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=str)
    parser.add_argument("--build_full_vrt", type=bool, default=True)
    params = parser.parse_args()
    p = params.p
    build_full_vrt = params.build_full_vrt

    if p is None:
        raise ValueError("Please specify path to folder --p")

    vrt_dir = Path(p) / "vrts"
    vrt_dir.mkdir(parents=True, exist_ok=True)
    tif_dir = Path(p) / "tifs"
    tif_dir.mkdir(parents=True, exist_ok=True)

    print("Building vrt for each batch")
    for d in glob(p + "/*/"):

        if "batch" not in d:
            continue

        match = re.search("batch_(.*?)-", d)
        if match:
            i = int(match.group(1))
        else:
            raise ValueError(f"Cannot parse i from {d}")
        vrt_file = Path(f"{vrt_dir}/{i}.vrt")
        if not vrt_file.exists():
            gdal_cmd(cmd_type="gdalbuildvrt", in_file=f"{d}*", out_file=str(vrt_file))

        if not build_full_vrt:
            tif_file = Path(f"{tif_dir}/{i}.tif")
            if not tif_file.exists():
                gdal_cmd(cmd_type="gdal_translate", in_file=str(vrt_file), out_file=str(tif_file))

    if build_full_vrt:
        gdal_cmd(
            cmd_type="gdalbuildvrt",
            in_file=f"{vrt_dir}/*.vrt",
            out_file=f"{vrt_dir}/final.vrt",
            msg="Building full vrt",
        )
        gdal_cmd(
            cmd_type="gdal_translate",
            in_file=f"{vrt_dir}/final.vrt",
            out_file=f"{tif_dir}/final.tif",
            msg="Vrt to tif",
        )
