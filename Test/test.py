import random
import argparse
import json
from typing import List

#  example run: python3 offline-test.py --n 50 --dist heavy --seed 42 --save test_items.json

def generate_uniform_items(n: int) -> List[float]:
    """Generate n items with sizes uniformly distributed in (0, 1]."""
    return [round(random.uniform(0.1, 1.0), 4) for _ in range(n)]

def generate_normal_items(n: int, mean=0.5, std=0.2) -> List[float]:
    """Generate n items with sizes from a truncated normal distribution."""
    items = []
    while len(items) < n:
        val = random.gauss(mean, std)
        if 0.05 <= val <= 1.0:
            items.append(round(val, 4))
    return items

def generate_heavy_tail_items(n: int) -> List[float]:
    """Generate items with more small items and fewer large ones."""
    items = []
    for _ in range(n):
        r = random.random()
        if r < 0.7:
            items.append(round(random.uniform(0.05, 0.3), 4))
        elif r < 0.9:
            items.append(round(random.uniform(0.3, 0.7), 4))
        else:
            items.append(round(random.uniform(0.7, 1.0), 4))
    return items

def save_items_to_file(items: List[float], filename: str):
    with open(filename, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} items to '{filename}'.")

def main():
    parser = argparse.ArgumentParser(description="Offline Bin Packing Test Case Generator")
    parser.add_argument("--n", type=int, default=100, help="Number of items to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--dist", choices=["uniform", "normal", "heavy"], default="uniform", help="Distribution type")
    parser.add_argument("--save", type=str, default=None, help="Optional filename to save items as JSON")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.dist == "uniform":
        items = generate_uniform_items(args.n)
    elif args.dist == "normal":
        items = generate_normal_items(args.n)
    elif args.dist == "heavy":
        items = generate_heavy_tail_items(args.n)
    else:
        raise ValueError("Unknown distribution type.")

    print(f"\nGenerated {len(items)} items ({args.dist} distribution):\n")
    print(items)

    if args.save:
        save_items_to_file(items, args.save)

if __name__ == "__main__":
    main()
