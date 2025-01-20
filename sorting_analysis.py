import pandas as pd
import time
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# Quick Sort Implementation
def quick_sort(data, key):
    """
    Performs Quick Sort on a list of names based on the specified key.

    :param data: List of names to be sorted.
    :param key: The key to sort the data by.
    :return: Sorted list of names.
    """
    if len(data) <= 1:
        return data
    pivot = data[len(data) // 2]
    left = [x for x in data if x[key] < pivot[key]]
    middle = [x for x in data if x[key] == pivot[key]]
    right = [x for x in data if x[key] > pivot[key]]
    return quick_sort(left, key) + middle + quick_sort(right, key)

# Merge Sort Implementation
def merge_sort(data, key):
    """
    Performs Merge Sort on a list of names based on the specified key.

    :param data: List of names to be sorted.
    :param key: The key to sort the data by.
    :return: Sorted list of names.
    """
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = merge_sort(data[:mid], key)
    right = merge_sort(data[mid:], key)
    return merge(left, right, key)

def merge(left, right, key):
    """
    Helper function to merge two sorted lists.

    :param left: Left sorted list.
    :param right: Right sorted list.
    :param key: The key to merge the data by.
    :return: Merged sorted list.
    """
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i][key] < right[j][key]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Performance Measurement Function
def measure_performance(data, key, algorithm):
    """
    Measures the execution time and memory usage of a sorting algorithm.

    :param data: DataFrame containing the dataset to be sorted.
    :param key: The key to sort the data by.
    :param algorithm: Sorting algorithm to be measured.
    :return: Execution time (seconds) and memory usage (MiB).
    """
    data_list = data.to_dict('records')  # Convert DataFrame to list of dictionaries
    start_time = time.time()
    mem_usage = memory_usage((algorithm, (data_list, key)), max_iterations=1, interval=0.01)
    exec_time = time.time() - start_time
    return exec_time, max(mem_usage)

# Main Script
if __name__ == "__main__":
    data = pd.read_csv("Data.csv")  

    # Prepare datasets
    sorted_data = data.sort_values(by='Age').reset_index(drop=True)
    reverse_sorted_data = data.sort_values(by='Age', ascending=False).reset_index(drop=True)
    random_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    datasets = {
        "Sorted": sorted_data,
        "Reverse Sorted": reverse_sorted_data,
        "Random": random_data
    }

    results = {
        "Dataset Type": [],
        "Algorithm": [],
        "Execution Time (s)": [],
        "Memory Usage (MiB)": []
    }

    # Run tests for Quick Sort and Merge Sort
    for dataset_name, dataset in datasets.items():
        for algorithm_name, algorithm in [("Quick Sort", quick_sort), ("Merge Sort", merge_sort)]:
            exec_time, mem_usage = measure_performance(dataset, "Age", algorithm)
            results["Dataset Type"].append(dataset_name)
            results["Algorithm"].append(algorithm_name)
            results["Execution Time (s)"].append(exec_time)
            results["Memory Usage (MiB)"].append(mem_usage)

    # Convert results to a DataFrame
    performance_df = pd.DataFrame(results)

    # Display performance results
    print("Performance Metrics:")
    print(performance_df)

    # Plot Execution Time
    plt.figure(figsize=(10, 6))
    for algo in performance_df["Algorithm"].unique():
        algo_data = performance_df[performance_df["Algorithm"] == algo]
        plt.bar(algo_data["Dataset Type"], algo_data["Execution Time (s)"], label=algo)
    plt.title("Execution Time Comparison")
    plt.xlabel("Dataset Type")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.show()

    # Plot Memory Usage
    plt.figure(figsize=(10, 6))
    for algo in performance_df["Algorithm"].unique():
        algo_data = performance_df[performance_df["Algorithm"] == algo]
        plt.bar(algo_data["Dataset Type"], algo_data["Memory Usage (MiB)"], label=algo)
    plt.title("Memory Usage Comparison")
    plt.xlabel("Dataset Type")
    plt.ylabel("Memory Usage (MiB)")
    plt.legend()
    plt.show()
