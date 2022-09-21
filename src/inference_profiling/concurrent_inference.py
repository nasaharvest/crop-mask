# The given script sends in multiple http requests at a particular time which can be used to test
# the performance of the torchserve deployed on Cloud run or as a standalone docker container.

# For 10-100 requests, the container memory capacity on cloud run remains upto 50%.
# CPU performance is around 53% for 10 requests and around 80% for 1000 requests
# Container instance count 30 for 10 requests and around 139 for 1000 requests
# Container startup latency 2-3s for 10-1000 requests. Anamolous reading of 5s found once.
# Max reuest latency was found out to be 1.06 min.Anamolous reading of 7.91 minutes found once.


# Importing packages
import time
import sys
import os
import psutil
import requests
import numpy as np
from futures3.thread import ThreadPoolExecutor

# Defining the model url
url = "https://crop-mask-example-grxg7bzh2a-uc.a.run.app/predictions/Togo_crop-mask_2019_February"

# Defining the list of images to pass into the inference url
img_list = np.loadtxt('cloud_img_files.txt', dtype='str')

if len(sys.argv) < 2:
    img_list = []
else:
    img_list = list(map(str, sys.argv[1:]))


# Inference function to send a request to the source url.
def inference_service(img_file):
    img_file = '''gs://crop-mask-example-inference-tifs/Togo_crop-mask_2019_February/
                  min_lat=-1.63_min_lon=29.12_max_lat=4.3_max_lon
                  =35.18_dates=2019-02-01_2020-02-01_all/batch_0/'''+img_file
    response = requests.post(url, data={"uri" : img_file})
    return response.json()


if __name__ == "__main__":
    print("\n The process id is", os.getpid())
    # print("\n The parent process id of the child process is",os.getppid())
    start_time = time.time()

    # Creating a ThreadPool object to send multiple HTTP requests simultaneously
    with ThreadPoolExecutor(max_workers=len(img_list)) as pool:
        results = pool.map(inference_service, img_list)
    end_time = time.time()

    print(f'''For {len(img_list)} parallel requests,the program finished in {end_time - start_time}
              seconds''')
    print("The total memory consumed is",
          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB.")
    # print("The total memory usage is",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(list(results))
