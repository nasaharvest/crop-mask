# Importing packages
#import multiprocessing
import time
from tracemalloc import start
#import requests
#import logging
import csv
#from argparse import ArgumentParser
import sys
#import resource
import os
import psutil
import requests
from futures3.thread import ThreadPoolExecutor

# Defining the model url
url="https://crop-mask-example-grxg7bzh2a-uc.a.run.app/predictions/Togo_crop-mask_2019_February"

#Defining the list of images to pass into the inference url
#img_list=["10000000000-0000000000.tif","10000000000-0000000256.tif",#"10000000000-0000000512.tif","10000000000-0000000768.tif",
## "10000000000-0000001024.tif","10000000000-0000001280.tif","10000000000-0000001536.tif","10000000000-0000001792.tif",
# "10000000000-0000002048.tif","10000000000-0000002304.tif","10000000000-0000002560.tif","10000000000-0000002816.tif",
# "10000000000-0000003072.tif","10000000000-0000003328.tif","10000000000-0000003584.tif","10000000000-0000003840.tif",
# "10000000000-0000004096.tif","10000000000-0000004352.tif","10000000000-0000004608.tif","10000000000-0000004864.tif",
# "10000000000-0000005120.tif","10000000000-0000005376.tif","10000000000-0000005632.tif","10000000000-0000005888.tif",
 #"10000000000-0000006144.tif"]

if len(sys.argv)<2:
    img_list=[]
else:
    img_list=list(map(str,sys.argv[1:]))

#Inference function to send a request to the source url.
def inference_service(img_file):
    img_file="gs://crop-mask-example-inference-tifs/Togo_crop-mask_2019_February/min_lat=-1.63_min_lon=29.12_max_lat=4.3_max_lon=35.18_dates=2019-02-01_2020-02-01_all/batch_0/"+img_file
    response=requests.post(url,data={"uri":img_file})
    return response.json()

if __name__ == "__main__":
    print("\n The process id is",os.getpid())
    #print("\n The parent process id of the child process is",os.getppid())
    start_time=time.time()
    with ThreadPoolExecutor(max_workers=len(img_list)) as pool:
        results = pool.map(inference_service, img_list)
    end_time=time.time()

    # with open('cloudrun_logs.csv','a') as file:
    #    writer=csv.writer(file)
    #    data=[len(img_list),end_time-start_time,psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2]
    #    writer.writerow(data)

    print(f"For {len(img_list)} parallel requests,the program finished in {end_time-start_time} seconds")
    print("The total memory consumed is",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,"MB.")
    #print("The total memory usage is",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(list(results))
