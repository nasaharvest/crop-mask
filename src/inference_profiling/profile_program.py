# Import the required utilities
import psutil as ps
import time
from subprocess import PIPE
import sys
import os
import csv
from argparse import ArgumentParser
import numpy as np

#Creating a parser
parser=ArgumentParser()

parser.add_argument('--n',required=True,type=int,default=3)

args=parser.parse_args()

#img_list=list(map(str,sys.argv[1:]))
img_list=np.array(["00000000000-0000000000.tif","00000000000-0000000256.tif","00000000000-0000000512.tif","00000000000-0000000768.tif"
 "00000000000-0000001024.tif","00000000000-0000001280.tif","00000000000-0000001536.tif","00000000000-0000001792.tif"
 "00000000000-0000002048.tif","00000000000-0000002304.tif","00000000000-0000002560.tif","00000000000-0000002816.tif",
 "00000000000-0000003072.tif","00000000000-0000003328.tif","00000000000-0000003584.tif","00000000000-0000003840.tif",
 "00000000000-0000004096.tif","00000000000-0000004352.tif","00000000000-0000004608.tif","00000000000-0000004864.tif",
 "00000000000-0000005120.tif","00000000256-0000000000.tif","00000000256-0000000256.tif","00000000256-0000000512.tif",
 "00000000256-0000000768.tif","00000000256-0000001024.tif","00000000256-0000001280.tif","00000000256-0000001536.tif",
 "00000000256-0000001792.tif","00000000256-0000002048.tif","00000000256-0000002304.tif","00000000256-0000002560.tif",
 "00000000000-0000004096.tif","00000000000-0000004352.tif","00000000000-0000004608.tif","00000000000-0000004864.tif"])

#n=int(sys.argv[1])
#define the command for the subprocess
img_fin=img_list[:args.n].tolist()
cmd = ["python3", "multiprocess_inference.py"]+img_fin

#Start the timer
start_time=time.time()

# Create the process
process = ps.Popen(cmd, stdout=PIPE)

print("The process ID is",process)

#Initialize the variables.
peak_sys_mem = 0
peak_sys_cpu = 0
peak_proc_mem = 0
#peak_proc_cpu = 0

# while the process is running calculate resource utilization.
print("Process is in the running state.")

while(process.is_running()):
    # set the sleep time to monitor at an interval of every second.
    time.sleep(1)
    #print(process.is_running())

    # capture the memory and cpu utilization at an instance
    sys_mem = ps.virtual_memory()[2]
    sys_cpu = ps.cpu_percent()
    proc_cpu=process.cpu_percent()
    proc_mem=process.memory_percent()
    # print("Memory calculated.")
    # print(proc_mem)
    # print("Process cpu")
    # print(proc_cpu)

    # track the peak utilization of the process
    if sys_mem > peak_sys_mem:
        peak_sys_mem = sys_mem
    if sys_cpu > peak_sys_cpu:
        peak_sys_cpu = sys_cpu
    
    # if proc_cpu>peak_proc_cpu:
    #         peak_proc_cpu = proc_cpu
    # if proc_mem>peak_proc_mem:
    #         peak_proc_mem = proc_mem
    if proc_mem == 0.0:
        break

    # Print the results to the monitor for each subprocess run.
end_time=time.time()

#Printing the results
print("For {} parallel requests the results are \n".format(args.n))
print(process.stdout.read())
#print("\n Peak process cpu usage is {} %".format(peak_proc_cpu))
#print("Peak process memory usage is {} %".format(peak_proc_mem))
print("The total time required for execution is",end_time-start_time)
print("\n Peak system memory usage is {} %".format(peak_sys_mem))
print("Peak system CPU utilization is {} %".format(peak_sys_cpu))

#Logging the results into a csv file
with open('cloudrun_service_logs.csv','a') as file:
    writer=csv.writer(file)
    if os.stat('cloudrun_service_logs.csv').st_size==0:
        writer.writerow(['No_of_requests','total_execution_time','Peak_cpu_percent','Peak_memory_percent'])
    data=[args.n,end_time-start_time,peak_sys_cpu,peak_sys_mem]
    writer.writerow(data)