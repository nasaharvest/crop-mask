from glob import glob
import os

if __name__ == "__main__":
    p = "/Users/izvonkov/nasaharvest/Uganda2"

    print("Building vrt for each batch")
    for i, d in enumerate(glob(p + "/*/")):
        cmd = f"gdalbuildvrt {p}/{i}.vrt {d}*"
        print(cmd)
        os.system(cmd)

    print("Building full vrt")
    cmd = f"gdalbuildvrt {p}/final.vrt {p}/*.vrt"
    print(cmd)
    os.system(cmd)

    print("Vrt to tif")
    cmd = f"gdal_translate -of GTiff {p}/final.vrt {p}/final.tif"
    print(cmd)
    os.system(cmd)
