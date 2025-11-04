from collections import defaultdict
import json
import time

def first_fit(items, capacity=1.0):
    bins = []
    for item in items:
        placed = False
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])  # Open new bin
    return bins

def best_fit(items, capacity=1.0):
    bins = []
    for item in items:
        best_bin_index = -1
        min_space_left = float('inf')
        for i, b in enumerate(bins):
            space_left = capacity - sum(b)
            if item <= space_left and space_left < min_space_left:
                best_bin_index = i
                min_space_left = space_left
        if best_bin_index >= 0:
            bins[best_bin_index].append(item)
        else:
            bins.append([item])  # Open new bin
    return bins

def next_fit(items, capacity=1.0):
    bins = []
    current_bin = []
    current_bin_sum = 0.0

    for item in items:
        if current_bin_sum + item <= capacity:
            current_bin.append(item)
            current_bin_sum += item
        else:
            bins.append(current_bin)
            current_bin = [item]
            current_bin_sum = item
    if current_bin:
        bins.append(current_bin)
    return bins

file_path = "../Test/test_items.json"
with open(file_path, 'r') as f:
    items = json.load(f)

start_time = time.time()
ff_result = first_fit(items)
end_time = time.time()
print("Packed bins (First Fit):")
for b in ff_result:
    print(b)
print(f"Total bins: {len(ff_result)}")
print(f"Time taken: {end_time - start_time:.4f} seconds\n")

start_time = time.time()
bf_result = best_fit(items)
end_time = time.time()
print("Packed bins (Best Fit):")
for b in bf_result:
    print(b)
print(f"Total bins: {len(bf_result)}")
print(f"Time taken: {end_time - start_time:.4f} seconds\n")

start_time = time.time()
nf_result = next_fit(items)
end_time = time.time()
print("Packed bins (Next Fit):")
for b in nf_result:
    print(b)
print(f"Total bins: {len(nf_result)}")
print(f"Time taken: {end_time - start_time:.4f} seconds")
