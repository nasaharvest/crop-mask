# This file used to measure performance parameters of the torchserve model deployed
# on the Docker container.For single models deployed on the Docker container,
# the system CPU percentage was found to reach 100% afer passing in 4-5 concurrent requests.
# More details can be found out in the Malawi_2020_September.csv and multi_model_logs.csv

# Import the required utilities

import psutil as ps
import time
from subprocess import PIPE
import os
import csv
from argparse import ArgumentParser
# import numpy as np
from utils import img_list

# Creating a parser
parser = ArgumentParser()

parser.add_argument('--n', required=True, type=int, default=3)

args = parser.parse_args()

# img_list = np.loadtxt('cloud_img_files.txt', dtype='str')

# define the command for the subprocess
img_fin = img_list[:args.n].tolist()
cmd = ["python3", "concurrent_inference.py"] + img_fin

# Start the timer
start_time = time.time()

# Create the process
process = ps.Popen(cmd, stdout=PIPE)

print("The process ID is", process)

# Initialize the variables.
peak_sys_mem = 0
peak_sys_cpu = 0
peak_proc_mem = 0

# while the process is running calculate resource utilization.
print("Process is in the running state.")

while (process.is_running()):
    # set the sleep time to monitor at an interval of every second.
    time.sleep(1)

    # capture the memory and cpu utilization at an instance
    sys_mem = ps.virtual_memory()[2]
    sys_cpu = ps.cpu_percent()
    proc_cpu = process.cpu_percent()
    proc_mem = process.memory_percent()

    # track the peak utilization of the process
    if sys_mem > peak_sys_mem:
        peak_sys_mem = sys_mem
    if sys_cpu > peak_sys_cpu:
        peak_sys_cpu = sys_cpu
    if proc_mem > peak_proc_mem:
        peak_proc_mem = proc_mem
    if proc_mem == 0.0:
        break

end_time = time.time()

# Printing the results
print("For {} parallel requests the results are \n".format(args.n))
print(process.stdout.read())
print("\n Peak system memory usage is {} %".format(peak_sys_mem))
print("Peak system CPU utilization is {} %".format(peak_sys_cpu))

total_run_time = end_time-start_time

# Logging the results into a csv file
with open('single_model_logs.csv', 'a') as file:
    writer = csv.writer(file)
    if os.stat('single_model_logs.csv').st_size == 0:
        writer.writerow(['No_of_requests', 'total_execution_time', 'Peak_cpu_percent',
                         'Peak_memory_percent', 'execution_time', 'Peak_process_memory'])
    data = [args.n, end_time-start_time, peak_sys_cpu, peak_sys_mem, total_run_time, peak_proc_mem]
    writer.writerow(data)
