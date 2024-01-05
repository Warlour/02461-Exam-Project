import multiprocessing
import os
import subprocess
import time

delay = 3 # Delay before starting next process in seconds
max_processes = 10 # Max number of concurrent processes
total = 40 # Total number of processes
finished = 0
current = 0

def worker(process):
    print(f"Worker {process}/{total} on PID {os.getpid()}")
    runargs = ["py", "main.py", 
        "-b 64", # Batch size
        #"-l 0.001", # Learning rate
        "-e 20", # Epochs
        "-m 0", # Momentum
        "-w 0.005", # Weight decay
        #"-s", # Disable scheduler
        "-o test18.xlsx" # Output filename
    ]
    if (process % max_processes) == 0:
        subprocess.run(runargs)
    else:
        subprocess.run(runargs, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    processes = []

    while finished < total:
        if len(processes) < max_processes and current < total:
            current += 1
            p = multiprocessing.Process(target=worker, args=(current,))
            p.start()
            processes.append(p)
            time.sleep(delay)  # Add delay before starting next process

        for p in processes:
            if not p.is_alive():
                processes.remove(p)
                finished += 1
                print("Finished worker", finished)

        if not processes:
            break
    
    print("Finished all processes")