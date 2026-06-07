import os
import subprocess
import random
import re
import time

# Configuration
DATASET_DIR = "Dataset_resized_256"
DB_PATH = "bird_cbir.db"
APP_PATH = "./cmake-build-debug/cbir_app"
NUM_QUERIES = 20
TOP_K = 5

def extract_species(filename):
    # Extracts species name from filename like "Acadian_Flycatcher_0005_29157.jpg"
    match = re.match(r"^([a-zA-Z_]+)_\d+", filename)
    if match:
        return match.group(1)
    return "Unknown"

def run_query(image_path):
    cmd = [APP_PATH, "--mode", "query", "--image", image_path, "--db", DB_PATH, "--topk", str(TOP_K)]
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        latency = (time.time() - start_time) * 1000 # ms
        return result.stdout, latency
    except subprocess.CalledProcessError as e:
        print(f"Error running query for {image_path}: {e.stderr}")
        return None, 0

def parse_results(stdout):
    results = []
    lines = stdout.strip().split('\n')
    for line in lines:
        if line.startswith("Top-"):
            continue
        # Format: 1. dist=0.221795 label=Dataset path=/Users/.../Dataset/Blue_Winged_Warbler_0017_161878.jpg
        match = re.search(r"path=(.+)$", line)
        if match:
            path = match.group(1)
            filename = os.path.basename(path)
            results.append(filename)
    return results

def main():
    if not os.path.exists(APP_PATH):
        print(f"Executable not found at {APP_PATH}. Please build the project first.")
        return

    all_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print(f"No images found in {DATASET_DIR}")
        return

    query_samples = random.sample(all_images, min(NUM_QUERIES, len(all_images)))
    
    total_precision = 0
    total_latency = 0
    successful_queries = 0

    print(f"{'#'*60}")
    print(f"{'Bird CBIR System Evaluation':^60}")
    print(f"{'#'*60}\n")
    print(f"{'Query Image':<40} | {'Precision@5':<12} | {'Latency (ms)':<12}")
    print("-" * 70)

    for query_filename in query_samples:
        query_path = os.path.join(DATASET_DIR, query_filename)
        query_species = extract_species(query_filename)
        
        stdout, latency = run_query(query_path)
        if stdout:
            results = parse_results(stdout)
            correct_count = 0
            for res_filename in results:
                res_species = extract_species(res_filename)
                if res_species == query_species:
                    correct_count += 1
            
            precision = (correct_count / TOP_K) * 100
            total_precision += precision
            total_latency += latency
            successful_queries += 1
            
            print(f"{query_filename[:39]:<40} | {precision:>10.1f}% | {latency:>10.1f}ms")
        else:
            print(f"{query_filename[:39]:<40} | {'FAILED':^12} | {'-':^12}")

    if successful_queries > 0:
        avg_precision = total_precision / successful_queries
        avg_latency = total_latency / successful_queries
        print("-" * 70)
        print(f"{'AVERAGE':<40} | {avg_precision:>10.1f}% | {avg_latency:>10.1f}ms")
        print("-" * 70)
        print(f"\nFinal Summary:")
        print(f"- Total Queries Run: {successful_queries}")
        print(f"- Mean Precision@5:  {avg_precision:.2f}%")
        print(f"- Mean Query Latency: {avg_latency:.2f} ms")
    else:
        print("No queries were successful.")

if __name__ == "__main__":
    main()
