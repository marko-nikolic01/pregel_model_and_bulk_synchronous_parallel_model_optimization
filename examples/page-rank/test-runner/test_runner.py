import subprocess
import os
import matplotlib.pyplot as plt

SEQUENTIAL_PAGE_RANK = "./page-rank/pageRankSequential"
PARALLEL_PAGE_RANK = "./page-rank/pageRankParallel"
DISTRIBUTED_PAGE_RANK = "./page-rank/pageRankDistributed"
ACCELERATED_PAGE_RANK = "./page-rank/pageRankAccelerated"

OUTPUT_DIR = "./output"

SUPERSTEPS_LIST = [10, 100, 1000, 10000, 100000]
MPI_PROCESSES = 4

def run_test(cmd, test_name):
    print(f"Running - {test_name}...", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print(f"{test_name} completed.\n", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed - {test_name}\n", flush=True)
        print(e, flush=True)

def read_execution_time(filepath):
    with open(filepath, "r") as f:
        return float(f.readline().strip())

def collect_execution_times(prefix):
    execution_times = []
    for supersteps in SUPERSTEPS_LIST:
        filename = f"{prefix}_{supersteps}.txt"
        path = os.path.join(OUTPUT_DIR, filename)
        execution_times.append(read_execution_time(path))
    return execution_times

def plot_execution_times(supersteps, sequential_execution_times, parallel_execution_times, distributed_execution_times, accelerated_execution_times):
    plt.figure()
    plt.plot(supersteps, sequential_execution_times, marker="o", label="Sequential", color="red")
    plt.plot(supersteps, parallel_execution_times, marker="o", label="Parallel (OpenMP)", color="green")
    plt.plot(supersteps, distributed_execution_times, marker="o", label="Distributed (OpenMPI)", color="blue")
    plt.plot(supersteps, accelerated_execution_times, marker="o", label="Accelerated (OpenCL)", color="yellow")

    plt.xlabel("Number of supersteps")
    plt.ylabel("Execution time (ms)")
    plt.title("PageRank Execution Time")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")

    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "execution_times.png"), dpi=300)
    plt.close()

def plot_speedups(supersteps, sequential_execution_times, parallel_execution_times, distributed_execution_times, accelerated_execution_times):
    parallel_speedups = [
        s / p for s, p in zip(sequential_execution_times, parallel_execution_times)
    ]
    distributed_speedups = [
        s / d for s, d in zip(sequential_execution_times, distributed_execution_times)
    ]
    accelerated_speedups = [
        s / a for s, a in zip(sequential_execution_times, accelerated_execution_times)
    ]

    plt.figure()
    plt.plot(supersteps, parallel_speedups, marker="o", label="Parallel (OpenMP)", color="green")
    plt.plot(supersteps, distributed_speedups, marker="o", label="Distributed (OpenMPI)", color="blue")
    plt.plot(supersteps, accelerated_speedups, marker="o", label="Accelerated (OpenCL)", color="yellow")

    plt.xlabel("Number of supersteps")
    plt.ylabel("Speedup")
    plt.title("PageRank Speedup")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")

    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "speed_ups.png"), dpi=300)
    plt.close()

def run_tests():
    # Run sequential tests
    for supersteps in SUPERSTEPS_LIST:
        run_test(
            [SEQUENTIAL_PAGE_RANK, str(supersteps)],
            f"Sequential ({supersteps} supersteps)"
        )

    # Run parallel tests
    for supersteps in SUPERSTEPS_LIST:
        run_test(
            [PARALLEL_PAGE_RANK, str(supersteps)],
            f"Parallel ({supersteps} supersteps)"
        )

    # Run distributed tests
    for supersteps in SUPERSTEPS_LIST:
        run_test(
            ["mpiexec", "--allow-run-as-root", "-n", str(MPI_PROCESSES), DISTRIBUTED_PAGE_RANK, str(supersteps)],
            f"Distributed ({supersteps} supersteps, {MPI_PROCESSES} processes)"
        )

    # Run accelerated tests
    for supersteps in SUPERSTEPS_LIST:
        run_test(
            [ACCELERATED_PAGE_RANK, str(supersteps)],
            f"Accelerated ({supersteps} supersteps)"
        )

    # Collect execution times
    sequential_execution_times = collect_execution_times("sequential")
    parallel_execution_times = collect_execution_times("parallel")
    distributed_execution_times = collect_execution_times("distributed")
    accelerated_execution_times = collect_execution_times("accelerated")

    # Generate plots
    plot_execution_times(SUPERSTEPS_LIST, sequential_execution_times, parallel_execution_times, distributed_execution_times, accelerated_execution_times)
    plot_speedups(SUPERSTEPS_LIST, sequential_execution_times, parallel_execution_times, distributed_execution_times, accelerated_execution_times)

if __name__ == "__main__":
    run_tests()
