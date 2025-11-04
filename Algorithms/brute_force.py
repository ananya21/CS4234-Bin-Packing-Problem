from itertools import product
from collections import defaultdict
import json
import time

def is_valid_assignment(assignment, items, capacity=1.0):
    bins = defaultdict(list)
    for idx, bin_id in enumerate(assignment):
        bins[bin_id].append(items[idx])
    for b in bins.values():
        if sum(b) > capacity:
            return False
    return True

def count_bins(assignment):
    return len(set(assignment))

def brute_force_offline_bin_packing(items):
    n = len(items)
    best_assignment = None
    min_bins = n 

    for assignment in product(range(n), repeat=n):
        if is_valid_assignment(assignment, items):
            num_bins = count_bins(assignment)
            if num_bins < min_bins:
                min_bins = num_bins
                best_assignment = assignment
                if min_bins == 1:
                    break

    bins = defaultdict(list)
    for idx, bin_id in enumerate(best_assignment):
        bins[bin_id].append(items[idx])
    
    return list(bins.values())

file_path = "../Test/test_items.json"
with open(file_path, 'r') as f:
    items = json.load(f)
    
start_time = time.time()
result = brute_force_offline_bin_packing(items)
end_time = time.time()
print("Packed bins (offline):")
for b in result:
    print(b)
    
print(f"\nTime taken: {end_time - start_time:.4f} seconds")
