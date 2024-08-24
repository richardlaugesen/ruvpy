from pathos.multiprocessing import ProcessPool as Pool
import numpy as np
import time


# CPU-intensive function: Multiple matrix multiplications
def very_heavy_computation(n):
    matrix_size = 200  # Large matrix size to increase complexity
    num_multiplications = 10  # Number of times to repeat the matrix multiplication

    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    # Perform multiple matrix multiplications in a loop
    for _ in range(num_multiplications):
        A = np.dot(A, B)

    # Return a simple summary to prevent optimization from skipping the computation
    return np.sum(A)


# Function to run the parallel test
def parallel_test(parallel_nodes, tasks, chunksize):
    # Start the parallel pool
    with Pool(nodes=parallel_nodes) as pool:
        # Map the very_heavy_computation function across the tasks with chunksize
        results = pool.map(very_heavy_computation, tasks, chunksize=chunksize)
    return results


if __name__ == "__main__":
    num_tasks = 100  # Number of tasks (adjust based on the test requirements)
    parallel_nodes = 6  # Number of parallel processes (matches the number of cores)
    tasks = list(range(num_tasks))  # Generate a list of tasks

    # Determine an appropriate chunksize
    chunksize = 1 #len(tasks) // parallel_nodes #max(1, len(tasks) // (10 * parallel_nodes))  # Adjust this factor as needed

    print(f"Running parallel test with {parallel_nodes} cores on {num_tasks} tasks, chunksize={chunksize}.")

    # Measure the time taken for parallel processing
    start_time = time.time()
    results = parallel_test(parallel_nodes, tasks, chunksize)
    end_time = time.time()

    # Output the results
    print(f"Results: {results}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
