import os
import sys

# Change the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if __name__ == "__main__":

    dest_asset = "users/izvonkov/crop-masks/Uganda_tile_1"
    src_uri = "gs://crop-mask-final/Uganda_April_2020_2021/1.tif"

    cmd_prefix = "earthengine upload image --asset_id="
