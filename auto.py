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
    runargs = ["py", "main.py", "-b 64", "-l 0.01",  "-e 20", "--weight_decay 0", "-ds", "-o sgd.xlsx"]
    if ((max_processes+1) % process) == 0 or process == 1:
        subprocess.run(runargs)
    else:
        subprocess.run(runargs, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    processes = []

    while finished < total:
        if len(processes) < max_processes:
            current += 1
            p = multiprocessing.Process(target=worker, args=(current,))
            p.start()
            processes.append(p)
            time.sleep(delay)  # Add delay before starting next process

        for p in processes:
            if not p.is_alive():
                processes.remove(p)
                finished += 1
                print("Finished process", current)

        if not processes:
            break