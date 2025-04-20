import atexit
import threading
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Optional

import ray

import dask
from dask.base import dont_optimize
from dask.core import istask, ishashable
from dask._task_spec import Task, Alias, TaskRef
from dask.system import CPU_COUNT
from dask.threaded import pack_exception, _thread_get_id

from ray.util.dask.callbacks import local_ray_callbacks, unpack_ray_callbacks
from ray.util.dask.common import unpack_object_refs
from ray.util.dask.scheduler_utils import get_async, apply_sync

main_thread = threading.current_thread()
default_pool = None
pools = defaultdict(dict)
pools_lock = threading.Lock()

TOP_LEVEL_RESOURCES_ERR_MSG = (
    'Use ray_remote_args={"resources": {...}} instead of resources={...} to specify '
    "required Ray task resources; see "
    "https://docs.ray.io/en/master/ray-core/package-ref.html#ray-remote."
)


def enable_dask_on_ray(
    shuffle: Optional[str] = "tasks",
    use_shuffle_optimization: Optional[bool] = True,
) -> dask.config.set:
    """
    Enable Dask-on-Ray scheduler. This helper sets the Dask-on-Ray scheduler
    as the default Dask scheduler in the Dask config. By default, it will also
    cause the task-based shuffle to be used for any Dask shuffle operations
    (required for multi-node Ray clusters, not sharing a filesystem), and will
    enable a Ray-specific shuffle optimization.

    >>> enable_dask_on_ray()
    >>> ddf.compute()  # <-- will use the Dask-on-Ray scheduler.

    If used as a context manager, the Dask-on-Ray scheduler will only be used
    within the context's scope.

    >>> with enable_dask_on_ray():
    ...     ddf.compute()  # <-- will use the Dask-on-Ray scheduler.
    >>> ddf.compute()  # <-- won't use the Dask-on-Ray scheduler.

    Args:
        shuffle: The shuffle method used by Dask, either "tasks" or
            "disk". This should be "tasks" if using a multi-node Ray cluster.
            Defaults to "tasks".
        use_shuffle_optimization: Enable our custom Ray-specific shuffle
            optimization. Defaults to True.
    Returns:
        The Dask config object, which can be used as a context manager to limit
        the scope of the Dask-on-Ray scheduler to the corresponding context.
    """
    if use_shuffle_optimization:
        from ray.util.dask.optimizations import dataframe_optimize
    else:
        dataframe_optimize = None
    # Manually set the global Dask scheduler config.
    # We also force the task-based shuffle to be used since the disk-based
    # shuffle doesn't work for a multi-node Ray cluster that doesn't share
    # the filesystem.
    return dask.config.set(
        scheduler=ray_dask_get, shuffle=shuffle, dataframe_optimize=dataframe_optimize, array_optimize=dont_optimize
    )


def disable_dask_on_ray():
    """
    Unsets the scheduler, shuffle method, and DataFrame optimizer.
    """
    return dask.config.set(scheduler=None, shuffle=None, dataframe_optimize=None)


def ray_dask_get(dsk, keys, **kwargs):
    """
    A Dask-Ray scheduler. This scheduler will send top-level (non-inlined) Dask
    tasks to a Ray cluster for execution. The scheduler will wait for the
    tasks to finish executing, fetch the results, and repackage them into the
    appropriate Dask collections. This particular scheduler uses a threadpool
    to submit Ray tasks.

    This can be passed directly to `dask.compute()`, as the scheduler:

    >>> dask.compute(obj, scheduler=ray_dask_get)

    You can override the currently active global Dask-Ray callbacks (e.g.
    supplied via a context manager), the number of threads to use when
    submitting the Ray tasks, or the threadpool used to submit Ray tasks:

    >>> dask.compute(
            obj,
            scheduler=ray_dask_get,
            ray_callbacks=some_ray_dask_callbacks,
            num_workers=8,
            pool=some_cool_pool,
        )

    Args:
        dsk: Dask graph, represented as a task DAG dictionary.
        keys (List[str]): List of Dask graph keys whose values we wish to
            compute and return.
        ray_callbacks (Optional[list[callable]]): Dask-Ray callbacks.
        num_workers (Optional[int]): The number of worker threads to use in
            the Ray task submission traversal of the Dask graph.
        pool (Optional[ThreadPool]): A multiprocessing threadpool to use to
            submit Ray tasks.

    Returns:
        Computed values corresponding to the provided keys.
    """
    num_workers = kwargs.pop("num_workers", None)
    pool = kwargs.pop("pool", None)
    # We attempt to reuse any other thread pools that have been created within
    # this thread and with the given number of workers. We reuse a global
    # thread pool if num_workers is not given and we're in the main thread.
    global default_pool
    thread = threading.current_thread()
    if pool is None:
        with pools_lock:
            if num_workers is None and thread is main_thread:
                if default_pool is None:
                    default_pool = ThreadPool(CPU_COUNT)
                    atexit.register(default_pool.close)
                pool = default_pool
            elif thread in pools and num_workers in pools[thread]:
                pool = pools[thread][num_workers]
            else:
                pool = ThreadPool(num_workers)
                atexit.register(pool.close)
                pools[thread][num_workers] = pool

    ray_callbacks = kwargs.pop("ray_callbacks", None)
    persist = kwargs.pop("ray_persist", False)
    enable_progress_bar = kwargs.pop("_ray_enable_progress_bar", None)

    # Handle Ray remote args and resource annotations.
    if "resources" in kwargs:
        raise ValueError(TOP_LEVEL_RESOURCES_ERR_MSG)
    ray_remote_args = kwargs.pop("ray_remote_args", {})
    try:
        annotations = dask.config.get("annotations")
    except KeyError:
        annotations = {}
    if "resources" in annotations:
        raise ValueError(TOP_LEVEL_RESOURCES_ERR_MSG)

    scoped_ray_remote_args = _build_key_scoped_ray_remote_args(
        dsk, annotations, ray_remote_args
    )

    with local_ray_callbacks(ray_callbacks) as ray_callbacks:
        # Unpack the Ray-specific callbacks.
        (
            ray_presubmit_cbs,
            ray_postsubmit_cbs,
            ray_pretask_cbs,
            ray_posttask_cbs,
            ray_postsubmit_all_cbs,
            ray_finish_cbs,
        ) = unpack_ray_callbacks(ray_callbacks)
        # NOTE: We hijack Dask's `get_async` function, injecting a different
        # task executor.
        object_refs = get_async(
            _apply_async_wrapper(
                pool.apply_async,
                _rayify_task_wrapper,
                ray_presubmit_cbs,
                ray_postsubmit_cbs,
                ray_pretask_cbs,
                ray_posttask_cbs,
                scoped_ray_remote_args,
            ),
            len(pool._pool),
            dsk,
            keys,
            get_id=_thread_get_id,
            pack_exception=pack_exception,
            **kwargs,
        )
        if ray_postsubmit_all_cbs is not None:
            for cb in ray_postsubmit_all_cbs:
                cb(object_refs, dsk)
        # NOTE: We explicitly delete the Dask graph here so object references
        # are garbage-collected before this function returns, i.e. before all
        # Ray tasks are done. Otherwise, no intermediate objects will be
        # cleaned up until all Ray tasks are done.
        del dsk
        if persist:
            result = object_refs
        else:
            pb_actor = None
            if enable_progress_bar:
                pb_actor = ray.get_actor("_dask_on_ray_pb")
            result = ray_get_unpack(object_refs, progress_bar_actor=pb_actor)
        if ray_finish_cbs is not None:
            for cb in ray_finish_cbs:
                cb(result)

    # cleanup pools associated with dead threads.
    with pools_lock:
        active_threads = set(threading.enumerate())
        if thread is not main_thread:
            for t in list(pools):
                if t not in active_threads:
                    for p in pools.pop(t).values():
                        p.close()
    return result


def _apply_async_wrapper(apply_async, real_func, *extra_args, **extra_kwargs):
    """
    Wraps the given pool `apply_async` function, hotswapping `real_func` in as
    the function to be applied and adding `extra_args` and `extra_kwargs` to
    `real_func`'s call.

    Args:
        apply_async: The pool function to be wrapped.
        real_func: The real function that we wish the pool apply
            function to execute.
        *extra_args: Extra positional arguments to pass to the `real_func`.
        **extra_kwargs: Extra keyword arguments to pass to the `real_func`.

    Returns:
        A wrapper function that will ignore it's first `func` argument and
        pass `real_func` in its place. To be passed to `dask.local.get_async`.
    """

    def wrapper(func, args=(), kwds=None, callback=None):  # noqa: M511
        if not kwds:
            kwds = {}
        return apply_async(
            real_func,
            args=args + extra_args,
            kwds=dict(kwds, **extra_kwargs),
            callback=callback,
        )

    return wrapper


def _rayify_task_wrapper(
    key,
    task_info,
    dumps,
    loads,
    get_id,
    pack_exception,
    ray_presubmit_cbs,
    ray_postsubmit_cbs,
    ray_pretask_cbs,
    ray_posttask_cbs,
    scoped_ray_remote_args,
):
    """
    The core Ray-Dask task execution wrapper, to be given to the thread pool's
    `apply_async` function. Exactly the same as `execute_task`, except that it
    calls `_rayify_task` on the task instead of `_execute_task`.

    Args:
        key: The Dask graph key whose corresponding task we wish to
            execute.
        task_info: The task to execute and its dependencies.
        dumps: A result serializing function.
        loads: A task_info deserializing function.
        get_id: An ID generating function.
        pack_exception: An exception serializing function.
        ray_presubmit_cbs: Pre-task submission callbacks.
        ray_postsubmit_cbs: Post-task submission callbacks.
        ray_pretask_cbs: Pre-task execution callbacks.
        ray_posttask_cbs: Post-task execution callbacks.
        scoped_ray_remote_args: Ray task options for each key.

    Returns:
        A 3-tuple of the task's key, a literal or a Ray object reference for a
        Ray task's result, and whether the Ray task submission failed.
    """
    try:
        task, deps = loads(task_info)
        result = _rayify_task(
            task,
            key,
            deps,
            ray_presubmit_cbs,
            ray_postsubmit_cbs,
            ray_pretask_cbs,
            ray_posttask_cbs,
            scoped_ray_remote_args.get(key, {}),
        )
        id = get_id()
        result = dumps((result, id))
        failed = False
    except BaseException as e:
        result = pack_exception(e, dumps)
        failed = True
    return key, result, failed


def _rayify_task(
    task,
    key,
    deps,
    ray_presubmit_cbs,
    ray_postsubmit_cbs,
    ray_pretask_cbs,
    ray_posttask_cbs,
    ray_remote_args,
):
    """
    Rayifies the given task, submitting it as a Ray task to the Ray cluster.

    Args:
        task: A Dask graph value, being either a literal, dependency
            key, Dask task, or a list thereof.
        key: The Dask graph key for the given task.
        deps: The dependencies of this task.
        ray_presubmit_cbs: Pre-task submission callbacks.
        ray_postsubmit_cbs: Post-task submission callbacks.
        ray_pretask_cbs: Pre-task execution callbacks.
        ray_posttask_cbs: Post-task execution callbacks.
        ray_remote_args: Ray task options. See :func:`ray.remote` for details.

    Returns:
        A literal, a Ray object reference representing a submitted task, or a
        list thereof.
    """
    if isinstance(task, list):
        # Recursively rayify this list. This will still bottom out at the first
        # actual task encountered, inlining any tasks in that task's arguments.
        return [
            _rayify_task(
                t,
                key,
                deps,
                ray_presubmit_cbs,
                ray_postsubmit_cbs,
                ray_pretask_cbs,
                ray_posttask_cbs,
                ray_remote_args,
            )
            for t in task
        ]
    elif istask(task):
        # Unpacks and repacks Ray object references and submits the task to the
        # Ray cluster for execution.
        if ray_presubmit_cbs is not None:
            alternate_returns = [cb(task, key, deps) for cb in ray_presubmit_cbs]
            for alternate_return in alternate_returns:
                # We don't submit a Ray task if a presubmit callback returns
                # a non-`None` value, instead we return said value.
                # NOTE: This returns the first non-None presubmit callback
                # return value.
                if alternate_return is not None:
                    return alternate_return

        if isinstance(task, Alias):
            target = task.target
            if isinstance(target, TaskRef):
                return deps[task.target.key]
            else:
                breakpoint()
        elif isinstance(task, Task):
            func, args = task.func, task.args
        else:
            # Fallback if task is a legacy graph.
            func, args = task[0], task[1:]
        if func is multiple_return_get:
            return _execute_task(task, deps)
        # If the function's arguments contain nested object references, we must
        # unpack said object references into a flat set of arguments so that
        # Ray properly tracks the object dependencies between Ray tasks.
        arg_object_refs, repack = unpack_object_refs(args, deps)
        # Submit the task using a wrapper function.
        object_refs = dask_task_wrapper.options(
            name=f"dask:{key!s}",
            num_returns=(
                1 if not isinstance(func, MultipleReturnFunc) else func.num_returns
            ),
            **ray_remote_args,
        ).remote(
            func,
            repack,
            key,
            ray_pretask_cbs,
            ray_posttask_cbs,
            *arg_object_refs,
        )

        if ray_postsubmit_cbs is not None:
            for cb in ray_postsubmit_cbs:
                cb(task, key, deps, object_refs)

        return object_refs
    elif not ishashable(task):
        return task
    elif task in deps:
        return deps[task]
    else:
        return task


def _execute_task(arg, cache, dsk=None):
    """Do the actual work of collecting data and executing a function

    Examples
    --------

    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> cache = {'x': 1, 'y': 2}

    Compute tasks against a cache
    >>> _execute_task((add, 'x', 1), cache)  # Compute task in naive manner
    2
    >>> _execute_task((add, (inc, 'x'), 1), cache)  # Support nested computation
    3

    Also grab data from cache
    >>> _execute_task('x', cache)
    1

    Support nested lists
    >>> list(_execute_task(['x', 'y'], cache))
    [1, 2]

    >>> list(map(list, _execute_task([['x', 'y'], ['y', 'x']], cache)))
    [[1, 2], [2, 1]]

    >>> _execute_task('foo', cache)  # Passes through on non-keys
    'foo'
    """
    if isinstance(arg, list):
        return [_execute_task(a, cache) for a in arg]
    elif istask(arg):
        func, args = arg[0], arg[1:]
        # Note: Don't assign the subtask results to a variable. numpy detects
        # temporaries by their reference count and can execute certain
        # operations in-place.
        return func(*(_execute_task(a, cache) for a in args))
    elif not ishashable(arg):
        return arg
    elif arg in cache:
        return cache[arg]
    else:
        return arg

from typing import Callable, Any
import numpy as np
from dask import config
from dask.core import flatten, reverse_dict
from dask.array.core import getter_inline
from dask.array.optimization import (
    fuse_slice,
    optimize,
    optimize_blockwise,
)
from dask.optimization import SubgraphCallable, fuse, inline_functions
from dask.highlevelgraph import HighLevelGraph
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.utils import ensure_dict
from dask.array.chunk import getitem
from dask.array.core import getter, getter_inline, getter_nofancy


# All get* functions the optimizations know about
GETTERS = (getter, getter_nofancy, getter_inline, getitem)
# These get* functions aren't ever completely removed from the graph,
# even if the index should be a no-op by numpy semantics. Some array-like's
# don't completely follow semantics, making indexing always necessary.
GETNOREMOVE = (getter, getter_nofancy)

def _is_getter_task(
    value,
) -> tuple[Callable, Any, Any, bool, bool | None] | None:
    """Check if a value in a Dask graph looks like a getter.
    1. Is it a tuple with the first element a known getter.
    2. Is it a SubgraphCallable with a single element in its
       dsk which is a known getter.
    If a getter is found, it returns a tuple with (getter, array, index, asarray, lock).
    Otherwise it returns ``None``.
    TODO: the second check is a hack to allow for slice fusion between tasks produced
    from blockwise layers and slicing operations. Once slicing operations have
    HighLevelGraph layers which can talk to Blockwise layers this check *should* be
    removed, and we should not have to introspect SubgraphCallables.
    """
    if type(value) is not tuple:
        return None
    first = value[0]
    get: Callable | None = None
    if first in GETTERS:
        get = first
    # We only accept SubgraphCallables with a single sub-task right now as it's
    # not clear which task to inspect if there is more than one, or how to resolve
    # conflicts if they occur.
    elif isinstance(first, SubgraphCallable) and len(first.dsk) == 1:
        v = next(iter(first.dsk.values()))
        if type(v) is tuple and len(v) > 1 and v[0] in GETTERS:
            get = v[0]
    if get is None:  # Didn't find a getter
        return None

    length = len(value)
    if length == 3:
        # getter defaults to asarray=True, getitem is semantically False
        return get, value[1], value[2], get is not getitem, None
    elif length == 5:
        return get, *value[1:]

    return None


def optimize_slices(dsk):
    """Optimize slices
    1.  Fuse repeated slices, like x[5:][2:6] -> x[7:11]
    2.  Remove full slices, like         x[:] -> x
    See also:
        fuse_slice_dict
    """
    fancy_ind_types = (list, np.ndarray)
    dsk = dsk.copy()
    for k, v in dsk.items():
        if a_task := _is_getter_task(v):
            get, a, a_index, a_asarray, a_lock = a_task

            while b_task := _is_getter_task(a):
                f2, b, b_index, b_asarray, b_lock = b_task

                if a_lock and a_lock is not b_lock:
                    break
                if (type(a_index) is tuple) != (type(b_index) is tuple):
                    break
                if type(a_index) is tuple:
                    indices = b_index + a_index
                    if len(a_index) != len(b_index) and any(i is None for i in indices):
                        break
                    if f2 is getter_nofancy and any(
                        isinstance(i, fancy_ind_types) for i in indices
                    ):
                        break
                elif f2 is getter_nofancy and (
                    type(a_index) in fancy_ind_types or type(b_index) in fancy_ind_types
                ):
                    break
                try:
                    c_index = fuse_slice(b_index, a_index)
                    # rely on fact that nested gets never decrease in
                    # strictness e.g. `(getter_nofancy, (getter, ...))` never
                    # happens
                    get = getter if f2 is getter_inline else f2
                except NotImplementedError:
                    break
                a, a_index, a_lock = b, c_index, b_lock
                a_asarray |= b_asarray

            # Skip the get call if not from from_array and nothing to do
            if get not in GETNOREMOVE and (
                (
                    type(a_index) is slice
                    and not a_index.start
                    and a_index.stop is None
                    and a_index.step is None
                )
                or (
                    type(a_index) is tuple
                    and all(
                        type(s) is slice
                        and not s.start
                        and s.stop is None
                        and s.step is None
                        for s in a_index
                    )
                )
            ):
                dsk[k] = a
            elif get is getitem or (a_asarray and not a_lock):
                # default settings are fine, drop the extra parameters Since we
                # always fallback to inner `get` functions, `get is getitem`
                # can only occur if all gets are getitem, meaning all
                # parameters must be getitem defaults.
                dsk[k] = (get, a, a_index)
            else:
                dsk[k] = (get, a, a_index, a_asarray, a_lock)

    return dsk

def optimize(
    dsk,
    keys,
    fuse_keys=None,
    fast_functions=None,
    inline_functions_fast_functions=(getter_inline,),
    rename_fused_keys=True,
    **kwargs,
):
    """Optimize dask for array computation
    1.  Cull tasks not necessary to evaluate keys
    2.  Remove full slicing, e.g. x[:]
    3.  Inline fast functions like getitem and np.transpose
    """
    if not isinstance(keys, (list, set)):
        keys = [keys]
    keys = list(flatten(keys))
    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
    dsk = optimize_blockwise(dsk, keys=keys)
    dsk = fuse_roots(dsk, keys=keys)
    dsk = dsk.cull(set(keys))
    # Perform low-level fusion unless the user has
    # specified False explicitly.
    if config.get("optimization.fuse.active") is False:
        return dsk

    dependencies = dsk.get_all_dependencies()
    dsk = ensure_dict(dsk)

    # Low level task optimizations
    if fast_functions is not None:
        inline_functions_fast_functions = fast_functions

    hold = hold_keys(dsk, dependencies)

    dsk, dependencies = fuse(
        dsk,
        hold + keys + (fuse_keys or []),
        dependencies,
        rename_keys=rename_fused_keys,
    )
    if inline_functions_fast_functions:
        dsk = inline_functions(
            dsk,
            keys,
            dependencies=dependencies,
            fast_functions=inline_functions_fast_functions,
        )

    return optimize_slices(dsk)


def hold_keys(dsk, dependencies):
    """Find keys to avoid fusion
    We don't want to fuse data present in the graph because it is easier to
    serialize as a raw value.
    We don't want to fuse chains after getitem/GETTERS because we want to
    move around only small pieces of data, rather than the underlying arrays.
    """
    dependents = reverse_dict(dependencies)
    data = {k for k, v in dsk.items() if type(v) not in (tuple, str)}

    hold_keys = list(data)
    for dat in data:
        deps = dependents[dat]
        for dep in deps:
            task = dsk[dep]
            # If the task is a get* function, we walk up the chain, and stop
            # when there's either more than one dependent, or the dependent is
            # no longer a get* function or an alias. We then add the final
            # key to the list of keys not to fuse.
            if _is_getter_task(task):
                try:
                    while len(dependents[dep]) == 1:
                        new_dep = next(iter(dependents[dep]))
                        new_task = dsk[new_dep]
                        # If the task is a get* or an alias, continue up the
                        # linear chain
                        if _is_getter_task(new_task) or new_task in dsk:
                            dep = new_dep
                        else:
                            break
                except (IndexError, TypeError):
                    pass
                hold_keys.append(dep)
    return hold_keys


@ray.remote
def dask_task_wrapper(func, repack, key, ray_pretask_cbs, ray_posttask_cbs, *args):
    """
    A Ray remote function acting as a Dask task wrapper. This function will
    repackage the given flat `args` into its original data structures using
    `repack`, execute any Dask subtasks within the repackaged arguments
    (inlined by Dask's optimization pass), and then pass the concrete task
    arguments to the provide Dask task function, `func`.

    Args:
        func: The Dask task function to execute.
        repack: A function that repackages the provided args into
            the original (possibly nested) Python objects.
        key: The Dask key for this task.
        ray_pretask_cbs: Pre-task execution callbacks.
        ray_posttask_cbs: Post-task execution callback.
        *args (ObjectRef): Ray object references representing the Dask task's
            arguments.

    Returns:
        The output of the Dask task. In the context of Ray, a
        dask_task_wrapper.remote() invocation will return a Ray object
        reference representing the Ray task's result.
    """
    if ray_pretask_cbs is not None:
        pre_states = [
            cb(key, args) if cb is not None else None for cb in ray_pretask_cbs
        ]
    # breakpoint()
    repacked_args, repacked_deps = repack(args)
    # Recursively execute Dask-inlined tasks.
    actual_args = [_execute_task(a, repacked_deps) for a in repacked_args]
    # Execute the actual underlying Dask task.
    result = func(*actual_args)

    if ray_posttask_cbs is not None:
        for cb, pre_state in zip(ray_posttask_cbs, pre_states):
            if cb is not None:
                cb(key, result, pre_state)

    return result


def render_progress_bar(tracker, object_refs):
    from tqdm import tqdm

    # At this time, every task should be submitted.
    total, finished = ray.get(tracker.result.remote())
    reported_finished_so_far = 0
    pb_bar = tqdm(total=total, position=0)
    pb_bar.set_description("")

    ready_refs = []

    while finished < total:
        submitted, finished = ray.get(tracker.result.remote())
        pb_bar.update(finished - reported_finished_so_far)
        reported_finished_so_far = finished
        ready_refs, _ = ray.wait(
            object_refs, timeout=0, num_returns=len(object_refs), fetch_local=False
        )
        if len(ready_refs) == len(object_refs):
            break
        import time

        time.sleep(0.1)
    pb_bar.close()
    submitted, finished = ray.get(tracker.result.remote())
    if submitted != finished:
        print("Completed. There was state inconsistency.")
    from pprint import pprint

    pprint(ray.get(tracker.report.remote()))


def ray_get_unpack(object_refs, progress_bar_actor=None):
    """
    Unpacks object references, gets the object references, and repacks.
    Traverses arbitrary data structures.

    Args:
        object_refs: A (potentially nested) Python object containing Ray object
            references.

    Returns:
        The input Python object with all contained Ray object references
        resolved with their concrete values.
    """

    def get_result(object_refs):
        if progress_bar_actor:
            render_progress_bar(progress_bar_actor, object_refs)
        return ray.get(object_refs)

    if isinstance(object_refs, tuple):
        object_refs = list(object_refs)

    if isinstance(object_refs, list) and any(
        not isinstance(x, ray.ObjectRef) for x in object_refs
    ):
        # We flatten the object references before calling ray.get(), since Dask
        # loves to nest collections in nested tuples and Ray expects a flat
        # list of object references. We repack the results after ray.get()
        # completes.
        object_refs, repack = unpack_object_refs(*object_refs)
        computed_result = get_result(object_refs)
        return repack(computed_result)
    else:
        return get_result(object_refs)


def ray_dask_get_sync(dsk, keys, **kwargs):
    """
    A synchronous Dask-Ray scheduler. This scheduler will send top-level
    (non-inlined) Dask tasks to a Ray cluster for execution. The scheduler will
    wait for the tasks to finish executing, fetch the results, and repackage
    them into the appropriate Dask collections. This particular scheduler
    submits Ray tasks synchronously, which can be useful for debugging.

    This can be passed directly to `dask.compute()`, as the scheduler:

    >>> dask.compute(obj, scheduler=ray_dask_get_sync)

    You can override the currently active global Dask-Ray callbacks (e.g.
    supplied via a context manager):

    >>> dask.compute(
            obj,
            scheduler=ray_dask_get_sync,
            ray_callbacks=some_ray_dask_callbacks,
        )

    Args:
        dsk: Dask graph, represented as a task DAG dictionary.
        keys (List[str]): List of Dask graph keys whose values we wish to
            compute and return.

    Returns:
        Computed values corresponding to the provided keys.
    """

    ray_callbacks = kwargs.pop("ray_callbacks", None)
    persist = kwargs.pop("ray_persist", False)

    with local_ray_callbacks(ray_callbacks) as ray_callbacks:
        # Unpack the Ray-specific callbacks.
        (
            ray_presubmit_cbs,
            ray_postsubmit_cbs,
            ray_pretask_cbs,
            ray_posttask_cbs,
            ray_postsubmit_all_cbs,
            ray_finish_cbs,
        ) = unpack_ray_callbacks(ray_callbacks)
        # NOTE: We hijack Dask's `get_async` function, injecting a different
        # task executor.
        object_refs = get_async(
            _apply_async_wrapper(
                apply_sync,
                _rayify_task_wrapper,
                ray_presubmit_cbs,
                ray_postsubmit_cbs,
                ray_pretask_cbs,
                ray_posttask_cbs,
            ),
            1,
            dsk,
            keys,
            **kwargs,
        )
        if ray_postsubmit_all_cbs is not None:
            for cb in ray_postsubmit_all_cbs:
                cb(object_refs, dsk)
        # NOTE: We explicitly delete the Dask graph here so object references
        # are garbage-collected before this function returns, i.e. before all
        # Ray tasks are done. Otherwise, no intermediate objects will be
        # cleaned up until all Ray tasks are done.
        del dsk
        if persist:
            result = object_refs
        else:
            result = ray_get_unpack(object_refs)
        if ray_finish_cbs is not None:
            for cb in ray_finish_cbs:
                cb(result)

        return result


@dataclass
class MultipleReturnFunc:
    func: callable
    num_returns: int

    def __call__(self, *args, **kwargs):
        returns = self.func(*args, **kwargs)
        if isinstance(returns, dict) or isinstance(returns, OrderedDict):
            returns = [returns[k] for k in range(len(returns))]
        return returns


def multiple_return_get(multiple_returns, idx):
    return multiple_returns[idx]


def _build_key_scoped_ray_remote_args(dsk, annotations, ray_remote_args):
    # Handle per-layer annotations.
    if not isinstance(dsk, dask.highlevelgraph.HighLevelGraph):
        dsk = dask.highlevelgraph.HighLevelGraph.from_collections(
            id(dsk), dsk, dependencies=()
        )
    # Build key-scoped annotations.
    scoped_annotations = {}
    layers = [(name, dsk.layers[name]) for name in dsk._toposort_layers()]
    for id_, layer in layers:
        layer_annotations = layer.annotations
        if layer_annotations is None:
            layer_annotations = annotations
        elif "resources" in layer_annotations:
            raise ValueError(TOP_LEVEL_RESOURCES_ERR_MSG)
        for key in layer.get_output_keys():
            layer_annotations_for_key = annotations.copy()
            # Layer annotations override global annotations.
            layer_annotations_for_key.update(layer_annotations)
            # Let same-key annotations earlier in the topological sort take precedence.
            layer_annotations_for_key.update(scoped_annotations.get(key, {}))
            scoped_annotations[key] = layer_annotations_for_key
    # Build key-scoped Ray remote args.
    scoped_ray_remote_args = {}
    for key, annotations in scoped_annotations.items():
        layer_ray_remote_args = ray_remote_args.copy()
        # Layer Ray remote args override global Ray remote args given in the compute
        # call.
        layer_ray_remote_args.update(annotations.get("ray_remote_args", {}))
        scoped_ray_remote_args[key] = layer_ray_remote_args
    return scoped_ray_remote_args
