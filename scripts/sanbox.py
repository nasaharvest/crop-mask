import rasterio
from rasterio.session import GSSession
from google.cloud import secretmanager

if __name__ == "__main__":
    gs_session = GSSession("/Users/izvonkov/Desktop/nasa-harvest-819cbbfe0587.json")
    file = "gs://ee-data-to-be-split/TestRegion_TestRegion_2020-04-01_2021-04-01.tif"
    with rasterio.Env(gs_session):
        with rasterio.open(file) as src:
            print(src.profile)



