from glob import glob
from typing import Optional
import os

build_full_vrt = False


def gdal_cmd(cmd_type: str, in_file: str, out_file: str, msg: Optional[str] = None):
    if cmd_type == "gdalbuildvrt":
        cmd = f"gdalbuildvrt {out_file} {in_file}"
    elif cmd_type == "gdal_translate":
        cmd = f"gdal_translate -a_srs EPSG:4326 -of GTiff {in_file} {out_file}"
    else:
        raise NotImplementedError(f"{cmd_type} not implemented.")
    if msg:
        print(msg)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    p = "/Users/izvonkov/nasaharvest/Uganda"

    print("Building vrt for each batch")
    for i, d in enumerate(glob(p + "/*/")):
        gdal_cmd(cmd_type="gdalbuildvrt", in_file=f"{d}*", out_file=f"{p}/{i}.vrt")
        gdal_cmd(cmd_type="gdal_translate", in_file=f"{p}/{i}.vrt", out_file=f"{p}/{i}.tif")

    if build_full_vrt:
        gdal_cmd(
            cmd_type="gdalbuildvrt",
            in_file=f"{p}/*.vrt",
            out_file=f"{p}/final.vrt",
            msg="Building full vrt",
        )
        gdal_cmd(
            cmd_type="gdal_translate",
            in_file=f"{p}/final.vrt",
            out_file=f"{p}/final.tif",
            msg="Vrt to tif",
        )
