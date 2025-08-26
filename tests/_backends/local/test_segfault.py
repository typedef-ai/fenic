from fenic import col, Session, SessionConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import sys
import traceback
import time

# number of rows
n = 10_000

# generate dictionary with 10 columns
data = {
    f"col{i}": np.random.randint(0, 1000, size=n).tolist()
    for i in range(10)
}

session = Session.get_or_create(
    SessionConfig(
        app_name="test_segfault",
    )
)
df = session.create_dataframe(data)
df.write.save_as_table("test_segfault", mode="overwrite")
print("Table created successfully")

def read_table(worker_id):
    print(f"Worker {worker_id} starting", flush=True)
    try:
        # Try creating a new session per thread
        # thread_session = Session.get_or_create(
        #     SessionConfig(app_name="test_segfault")
        # )
        # result = thread_session.table("test_segfault").to_pylist()

        result = session.table("test_segfault").to_pylist()
        print(f"Worker {worker_id} completed", flush=True)
        return result
    except Exception as e:
        print(f"Worker {worker_id} error: {e}", flush=True)
        traceback.print_exc()
        return None

# First test single-threaded
print('Testing single-threaded read...', flush=True)
try:
    result = read_table("single")
    print(f"Single-threaded success: {len(result)} rows", flush=True)
except Exception as e:
    print(f"Single-threaded failed: {e}", flush=True)
    sys.exit(1)

# Now test multi-threaded
print('Starting multi-threaded test...', flush=True)
try:
    with ThreadPoolExecutor(max_workers=2) as executor:  # Start with just 2 workers
        print("Submitting futures...", flush=True)
        futures = [executor.submit(read_table, i) for i in range(2)]
        print(f"Submitted {len(futures)} futures", flush=True)

        for i, future in enumerate(as_completed(futures)):
            print(f"Future {i} completed", flush=True)
            try:
                result = future.result(timeout=10)  # Add timeout
                if result is not None:
                    print(f"Got result with {len(result)} rows", flush=True)
            except Exception as e:
                print(f"Future {i} raised exception: {e}", flush=True)
                traceback.print_exc()
except Exception as e:
    print(f"ThreadPoolExecutor error: {e}", flush=True)
    traceback.print_exc()

print("Program completed", flush=True)
