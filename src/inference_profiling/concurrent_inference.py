# The given script sends in multiple http requests at a particular time which can be used to test
# the performance of the torchserve deployed on Cloud run or as a standalone docker container.

# For 10 requests, the container CPU utilization reached 32.33% and memory utilization
# reached 34.75%. For 100 concurrent requests, the container CPU utilization reached 82.23% and
#  the container memory utilzation reached 46.15%. For 1000  concurrent requests, the container
# CPU utilization reached 75.39%, and the container memory utilization reached 46.55%.
# More details about additional performance parameters can be found in the 'cloud_run_logs.txt'
# file.


import os
import sys

# Importing packages
import time

import requests
from futures3.thread import ThreadPoolExecutor

# Defining the model url
url = "https://crop-mask-example-grxg7bzh2a-uc.a.run.app/predictions/Togo_crop-mask_2019_February"

if len(sys.argv) < 2:
    img_list = []
else:
    img_list = list(map(str, sys.argv[1:]))


# Inference function to send a request to the source url.
def inference_service(img_file):
    img_url = """gs://crop-mask-example-inference-tifs/""" + img_file
    response = requests.post(url, data={"uri": img_url})
    return response.json()


if __name__ == "__main__":
    print("\n The process id is", os.getpid())
    start_time = time.time()

    # Creating a ThreadPool object to send multiple HTTP requests simultaneously
    with ThreadPoolExecutor(max_workers=len(img_list)) as pool:
        results = pool.map(inference_service, img_list)
    end_time = time.time()

    print(
        f"""For {len(img_list)} parallel requests,the program finished in {end_time - start_time}
              seconds"""
    )
    print(list(results))
