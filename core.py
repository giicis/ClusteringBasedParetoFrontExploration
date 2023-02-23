import copy
from abc import ABC, abstractmethod
from builtins import RuntimeError

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaders, to_tree


class ExplorerState:
    def __init__(self, indexes: np.ndarray, linkage: np.ndarray, clusters: np.ndarray | None = None, filters: np.ndarray | None = None, level: int = 0):
        if (len(indexes) - 1) != len(linkage):
            raise ValueError("While building Explorer State. Linkage is not adequate for these indexes."
                             "\nIndexes: {}\nLinkage: {}".format(indexes, linkage))
        if clusters is not None and len(indexes) != len(clusters):
            raise ValueError("While building Explorer State. Indexes and clusters must have the same length."
                             "\nIndexes: {}\n Clusters: {}".format(indexes, clusters))
        if filters is not None and len(indexes) != len(filters):
            raise ValueError("While building Explorer State. Indexes and filters must have the same length."
                             "\nIndexes: {}\nClusters: {}".format(indexes, filters))
        self._indexes = indexes
        self._linkage = linkage
        self._clusters = clusters
        self._filters = filters
        self._level = level

    def __copy__(self):
        return ExplorerState(self._indexes.copy(), self._linkage.copy(),
                             self._clusters.copy() if self._clusters is not None else None,
                             self._filters.copy() if self._filters is not None else None,
                             self._level)

    @property
    def linkage(self):
        return self._linkage

    # @linkage.setter
    # def linkage(self, nl):
    #    self._linkage = nl

    @property
    def indexes(self):
        return self._indexes if self._filters is None else self._indexes[self._filters]

    @property
    def clusters(self):
        return self._clusters

    @property
    def filters(self):
        return self._filters

    @property
    def level(self):
        return self._level


class ExplorerCommand(ABC):
    @abstractmethod
    def do(self, state: ExplorerState) -> ExplorerState:
        pass

    @abstractmethod
    def undo(self) -> ExplorerState:
        pass


class ExplorerCommandDefaultPersistence(ExplorerCommand):
    def __init__(self):
        self._state = None

    @abstractmethod
    def do(self, state: ExplorerState) -> ExplorerState:
        #self._state = copy.copy(state)
        self._state = state
        return self._state

    def undo(self) -> ExplorerState:
        if self._state is None:
            raise RuntimeError("It is not possible to undo a command that has not yet been done.")
        return self._state


class ClusterCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, nclusters: int):
        super().__init__()
        if nclusters <= 1:
            raise ValueError(
                "Error while clustering. The number of clusters must be > 1. {} received.".format(nclusters))
        self._nclusters = nclusters

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        clusters = fcluster(state.linkage, self._nclusters, criterion='maxclust')
        return ExplorerState(state.indexes, state.linkage, clusters, level=state.level)


class ZoomInCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, selected_cluster: int):
        if selected_cluster <= 0:
            raise ValueError("While building ZoomInCommand. selected_cluster must be a positive integer ({} received)".format(selected_cluster))
        super().__init__()
        self._selected_cluster = selected_cluster

    def filter_linkage(self, linkage, node):
        return [] if node.is_leaf() else self.filter_linkage(linkage, node.left) \
                                         + self.filter_linkage(linkage, node.right) \
                                         + [(node.id, list(linkage[node.id - len(linkage) - 1]))]

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        clusters = state.clusters
        if clusters is None:
            raise ValueError("Zoom in command cannot be executed in an unclustered state.")
        if self._selected_cluster > (NCLUSTERS := max(state.clusters)):
            raise ValueError("Zoom in command cannot be executed. Selected cluster is out of bounds. Number of clusters is {} and I want to select {}.".format(NCLUSTERS, self._selected_cluster))
        L, M = leaders(state.linkage, clusters)
        selected_root = L[np.where(M == self._selected_cluster)][0]
        tree = to_tree(state.linkage, rd=True)
        selected_tree = tree[1][selected_root]

        filtered_linkage = self.filter_linkage(state.linkage, selected_tree)

        leafs = selected_tree.pre_order(lambda x: x.id)
        filtered_indexes = np.array([state.indexes[x] for x in leafs])

        converted_ids = {leaf: i for i, leaf in enumerate(leafs)}
        converted_ids = converted_ids | {row[0]: (len(leafs) + i) for i, row in enumerate(filtered_linkage)}

        filtered_linkage_recalc_indexes = np.array(
            [[converted_ids[int(row[0])], converted_ids[int(row[1])], row[2], row[3]] for i, row in filtered_linkage])

        return ExplorerState(filtered_indexes, filtered_linkage_recalc_indexes, level=state.level+1)


class ApplyFiltersCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, keys: np.ndarray):
        super().__init__()
        self._keys = keys

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        if len(self._keys) != len(state.indexes):
            raise ValueError("ApplyFiltersCommand cannot be executed. Keys must be of same length as state's indexes.")
        return ExplorerState(state.indexes, state.linkage, state.clusters, self._keys, state.level)


class ParetoFrontExplorer:
    def __init__(self, state, done_commands=[], undone_commands=[]):
        self._actual_state = state
        self._done_commands = done_commands
        self._undone_commands = undone_commands

    def cluster(self, nclusters):
        pass

    def zoom_in(self, selected_cluster):
        pass

    def zoom_out(self):
        pass

    def restore(self):
        pass

    def apply_filter(self):
        pass