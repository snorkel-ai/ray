import ray
from ray.util.dask import ray_dask_get, enable_dask_on_ray, disable_dask_on_ray
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

# Start Ray.
# Tip: If connecting to an existing cluster, use ray.init(address="auto").
ray.init(runtime_env={
    "env_vars": {"RAY_DEBUG": "1"}, 
})

d_arr = da.from_array(np.random.randint(0, 1000, size=(1024, 1024)))
d_arr2 = da.from_array(np.random.randint(0, 1000, size=(1024, 1024)))

res = d_arr.compute(scheduler=ray_dask_get)
print("array: ", res)
res = d_arr2.compute(scheduler=ray_dask_get)
print("array2: ", res)
res = (d_arr + d_arr2).compute(scheduler=ray_dask_get)
print("array + array2: ", res)
res = d_arr.mean().compute(scheduler=ray_dask_get)
print("mean of array: ", res)

enable_dask_on_ray()

npartitions = 2
df = dd.from_pandas(
    pd.DataFrame(np.random.randint(0, 100, size=(1024, 2)), columns=["age", "grade"]),
    npartitions=npartitions
)
# df.visualize doesn't work with dask-expr
#df.visualize(filename="df.png", optimize_graph=False)

df2 = dd.from_pandas(
    pd.DataFrame(np.random.randint(0, 100, size=(1024, 2)), columns=["age", "grade"]),
    npartitions=npartitions
)
# We set max_branch=npartitions in order to ensure that the task-based
# shuffle happens in a single stage, which is required in order for our
# optimization to work.
a = df.set_index(["age"], shuffle="tasks", max_branch=npartitions)
b = df2.set_index(["age"], shuffle="tasks", max_branch=npartitions)

#a.visualize(filename="a.png", optimize_graph=False)
#(dd.concat([a, b])).visualize(filename="a_plus_b.png", optimize_graph=False)
print("ddf a: ", a.compute())
print("ddf a + ddf b: ", dd.concat([a, b]).compute())

disable_dask_on_ray()

ray.shutdown()
