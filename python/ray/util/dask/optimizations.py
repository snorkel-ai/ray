import operator
import warnings

import dask
from dask import core
from dask.core import istask
from dask.dataframe.core import _concat
from dask.highlevelgraph import HighLevelGraph

from .scheduler import MultipleReturnFunc, multiple_return_get

try:
    from dask.dataframe.dask_expr._shuffle import SimpleShuffle
except ImportError:
    SimpleShuffle = None

if SimpleShuffle is not None:

    class MultipleReturnSimpleShuffle(SimpleShuffle):
        def __str__(self):
            return (
                f"MultipleReturnSimpleShuffle<name='{self._name[-7:]}', "
                f"npartitions={self.npartitions_out}>"
            )

        # def __reduce__(self):
        #     attrs = [
        #         "name",
        #         "column",
        #         "npartitions",
        #         "npartitions_input",
        #         "ignore_index",
        #         "name_input",
        #         "meta_input",
        #         "parts_out",
        #         "annotations",
        #     ]
        #     return (
        #         MultipleReturnSimpleShuffleLayer,
        #         tuple(getattr(self, attr) for attr in attrs),
        #     )

        # def _cull(self, parts_out):
        #     return MultipleReturnSimpleShuffleLayer(
        #         self.name,
        #         self.column,
        #         self.npartitions,
        #         self.npartitions_input,
        #         self.ignore_index,
        #         self.name_input,
        #         self.meta_input,
        #         parts_out=parts_out,
        #     )

        def _layer(self):
            """Construct graph for a simple shuffle operation."""
            shuffle_group_name = "group-" + self._name
            split_name = "split-" + self._name
            npartitions = self.npartitions_out

            dsk = {}
            n_parts_out = len(self._partitions)
            _filter = self._partitions if self._filtered else None
            for global_part, part_out in enumerate(self._partitions):
                _concat_list = [
                    (split_name, part_out, part_in)
                    for part_in in range(self.frame.npartitions)
                ]
                dsk[(self._name, global_part)] = (
                    _concat,
                    _concat_list,
                    self.ignore_index,
                )
                for _, _part_out, _part_in in _concat_list:
                    dsk[(split_name, _part_out, _part_in)] = (
                        multiple_return_get,
                        (shuffle_group_name, _part_in),
                        _part_out,
                    )
                    if (shuffle_group_name, _part_in) not in dsk:
                        dsk[(shuffle_group_name, _part_in)] = (
                            MultipleReturnFunc(
                                self._shuffle_group,
                                n_parts_out,
                            ),
                            (self.frame._name, _part_in),
                            _filter,
                            self.partitioning_index,
                            0,
                            npartitions,
                            npartitions,
                            self.ignore_index,
                            npartitions,
                        )

            return dsk

    def rewrite_simple_shuffle_layer(dsk, keys):
        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
        else:
            dsk = dsk.copy()

        layers = dsk.layers.copy()
        for key, layer in layers.items():
            if type(layer) is SimpleShuffleLayer:
                dsk.layers[key] = MultipleReturnSimpleShuffleLayer.clone(layer)
        return dsk

    def dataframe_optimize(dsk, keys, **kwargs):
        if not isinstance(keys, (list, set)):
            keys = [keys]
        keys = list(core.flatten(keys))

        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())

        dsk = rewrite_simple_shuffle_layer(dsk, keys=keys)
        return dsk

else:

    def dataframe_optimize(dsk, keys, **kwargs):
        warnings.warn(
            "Custom dataframe shuffle optimization only works on "
            "dask>=2020.12.0, you are on version "
            f"{dask.__version__}, please upgrade Dask."
            "Falling back to default dataframe optimizer."
        )
        return dsk


# Stale approaches below.


def fuse_splits_into_multiple_return(dsk, keys):
    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
    else:
        dsk = dsk.copy()
    dependencies = dsk.dependencies.copy()
    for k, v in dsk.items():
        if istask(v) and v[0] == shuffle_group:
            task_deps = dependencies[k]
            # Only rewrite shuffle group split if all downstream dependencies
            # are splits.
            if all(
                istask(dsk[dep]) and dsk[dep][0] == operator.getitem
                for dep in task_deps
            ):
                for dep in task_deps:
                    # Rewrite split
                    pass
