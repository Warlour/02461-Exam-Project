import multiprocessing
import os
import subprocess
import time


class AutoRunner:
    def __init__(self, delay, max_processes, total, runargs) -> None:
        self.processes = []
        self.delay = delay # Delay before starting next process in seconds
        self.max_processes = max_processes # Max number of concurrent processes
        self.total = total # Total number of processes
        self.finished = 0
        self.current = 0
        self.runargs = runargs
    
    @staticmethod
    def worker(process, total, max_processes, runargs) -> None:
        print(f"Worker {process}/{total} on PID {os.getpid()}")
        runargs = ["py", "main.py"] + runargs
            
        if (process % max_processes) == 0:
            subprocess.run(runargs)
            print(runargs)
        else:
            subprocess.run(runargs, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    def run(self) -> None:
        while self.finished < self.total:
            if len(self.processes) < self.max_processes and self.current < self.total:
                self.current += 1
                p = multiprocessing.Process(target=self.worker, args=(self.current, self.total, self.max_processes, self.runargs))
                p.start()
                self.processes.append(p)
                time.sleep(self.delay)
            
            for p in self.processes:
                if not p.is_alive():
                    self.processes.remove(p)
                    self.finished += 1
                    print("Finished worker", self.finished)
            
            if not self.processes:
                break

        print("Finished all processes")
        
if __name__ == "__main__":
    runargs = [
        "--batch_size",         "64", # Batch size
        "--learning_rate",      "0.01", # Learning rate 
        "--epochs",             "20", # Epochs
        "--momentum",           "0", # Momentum
        "--weight_decay",       "0", # Weight decay
        "--scheduler_type",     "None", # Scheduler
        "--gamma",              "0.5", # Gamma
        "--min_lr",             "0", # Minimum learning rate
        "--output_csv",         "test1.xlsx" # Output filename
        "--note",               "Test 1"
    ]
    runner = AutoRunner(delay=3, max_processes=10, total=40, runargs=runargs)
    runner.run()
    