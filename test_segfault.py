import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from fenic import Session, SessionConfig

# number of rows
n = 10_000

# generate dictionary with 10 columns
data = {
    f"col{i}": np.random.randint(0, 100000, size=n).tolist()
    for i in range(10)
}

session = Session.get_or_create(
    SessionConfig(
        app_name="test_segfault",
    )
)
df = session.create_dataframe(data)
df.write.save_as_table("test_segfault", mode="overwrite")
df.write.save_as_table("test_segfault_2", mode="overwrite")
df.write.save_as_table("test_segfault_3", mode="overwrite")


def read_table():
    thread_id = threading.current_thread().ident

    # Randomly pick one of the three tables
    table_name = random.choice(["test_segfault", "test_segfault_2", "test_segfault_3"])
    print(f"Thread {thread_id} starting - reading table: {table_name}")

    # Create a new session for each thread - this is the fix!
    thread_session = Session.get_or_create(
        SessionConfig(
            app_name="test_segfault",
        )
    )
    result = thread_session.table(table_name).to_pylist()

    print(f"Thread {thread_id} completed read of {table_name}")
    return result, table_name


print('concurrent starting')
with ThreadPoolExecutor(max_workers=1000) as executor:
    futures = [executor.submit(read_table) for _ in range(1000)]
    for i, future in enumerate(as_completed(futures)):
        thread_id = threading.current_thread().ident
        print(f"Main thread {thread_id} processing future {i}")
        result = future.result()
        print(f"Future {i} finished. Got {len(result[0])} rows from {result[1]}")
print('concurrent finished')

print('sequential starting')
with ThreadPoolExecutor(max_workers=1000) as executor:
    for i in range(1000):
        # Submit one task at a time and wait for it to complete
        future = executor.submit(read_table)
        thread_id = threading.current_thread().ident
        print(f"Main thread {thread_id} processing task {i}")
        result = future.result()  # This blocks until the task completes
        print(f"Task {i} finished. Got {len(result[0])} rows from {result[1]}")
print('sequential finished')

print('done')
