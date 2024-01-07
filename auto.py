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
        "--batch_size",         "64", # Batch size
        "--learning_rate",      "0.01", # Learning rate
        "--epochs",             "20", # Epochs
        "--momentum",           "0", # Momentum
        "--weight_decay",       "0", # Weight decay
        "--scheduler_type",     "None", # Scheduler
        "--gamma",              "0.5", # Gamma
        "--min_lr",             "0", # Minimum learning rate
        "--output_csv",         "test1.xlsx" # Output filename
        "--note", "test1"
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