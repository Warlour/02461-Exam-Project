import multiprocessing
import os
import subprocess
import time

delay = 2 # Delay before starting next process in seconds
max_processes = 10 # Max number of concurrent processes
total = 40 # Total number of processes
launchedprocesses = 0
finishedprocesses = 0

def worker(process):
    print(f"Worker {process}/{total} on PID {os.getpid()}")
    runargs = ["py", "main.py", "-b 64", "-l 0.001",  "-e 20", "--disable_scheduler"]
    if (process % (max_processes+1)) == 0 or process == 1:
        subprocess.run(runargs)
    else:
        subprocess.run(runargs, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    processes = []

    while True:
        if len(processes) < max_processes:
            launchedprocesses += 1
            p = multiprocessing.Process(target=worker, args=(launchedprocesses,))
            p.start()
            processes.append(p)
            time.sleep(delay)  # Add delay before starting next process

        for p in processes:
            if not p.is_alive() and finishedprocesses < total:
                processes.remove(p)
                finishedprocesses += 1

        if not processes or finishedprocesses >= total:
            break