# This script is used to profile the performance of test_models.py file to determine the performance
# parameters for predictions made on pytorch models on a local environment. For more details check
# check out the script "test_models.py "

# Import the required utilities
import psutil as ps
import time
from subprocess import PIPE
import os
import csv
import sys


# img_list=list(map(str,sys.argv[1:]))

n = sys.argv[1]
model_name = sys.argv[2]
# define the command for the subprocess
cmd = ["python", "test_models.py"]+[n, model_name]
print(cmd)

# Start the timer
start_time = time.time()

# Create the process
process = ps.Popen(cmd, stdout=PIPE)

print("The process ID is", process)

# Initialize the variables.
peak_sys_mem = 0
peak_sys_cpu = 0
peak_proc_mem = 0
peak_proc_cpu = 0

# while the process is running calculate resource utilization.
print("Process is in the running state.")

while (process.is_running()):
    # set the sleep time to monitor at an interval of every second.
    time.sleep(0.5)

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
    if proc_cpu > peak_proc_cpu:
        peak_proc_cpu = proc_cpu
    if proc_mem == 0.0:
        break

    # Print the results to the monitor for each subprocess run.
end_time = time.time()

# Printing the results
print("For batch size of {} the results are \n".format(n))
print(process.stdout.read())
print("\n Peak process cpu usage is {} %".format(peak_proc_cpu))
print("Peak process memory usage is {} %".format(peak_proc_mem))
print("The total time required for execution is", end_time-start_time)
print("\n Peak system memory usage is {} %".format(peak_sys_mem))
print("Peak system CPU utilization is {} %".format(peak_sys_cpu))

# Logging the results into a csv file
with open('local_model_tests.csv', 'a') as file:
    writer = csv.writer(file)
    if os.stat('windows_logs.csv').st_size == 0:
        writer.writerow(['Model_name', 'Batch_size', 'total_execution_time', 'Peak_cpu_percent',
                         'Peak_memory_percent', 'peak_proc_cpu', 'peak_proc_mem'])
    data = [model_name, n, end_time-start_time, peak_sys_cpu, peak_sys_mem, peak_proc_cpu,
            peak_proc_mem]
    writer.writerow(data)
