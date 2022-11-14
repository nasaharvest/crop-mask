# The given script creates multiple processes which in turn send multiple http requests
# at a particular time to test the performance of the torchserve deployed
#  on a standalone docker container. As the number of requests increase,
# the system CPU utilization was found to reach 100%. For the scenario wherein
# multiple models were deployed the average runtime was ~120ms.
# For single model deployed on torchserve, the execution time is ~50ms for upto 4 requests
# and then it rose to ~120ms on average for 5-10 requests. More details about the test results
# can be found in the "multi_models_logs.csv" and "single_model_logs.csv".


# Importing packages
import multiprocessing
import sys

import requests

# Defining the model url
url = "http://localhost:8080/predictions/Namibia_North_2020/"

img_list = sys.argv[1:]


# Inference function to send a request to the source url.
def inference_service(img_file):
    img_url = """gs://crop-mask-example-inference-tifs/""" + img_file
    response = requests.post(url, data={"uri": img_url})
    return response.json()


if __name__ == "__main__":

    # Creating a pool object for spawning multiprocesses
    pool = multiprocessing.Pool(len(img_list))
    results = pool.map(inference_service, img_list)

    # Closing and joining the pool objects once the results are obtained
    pool.close()
    pool.join()

    # Printing out the results.
    print(results)
