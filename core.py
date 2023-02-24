#import copy
from abc import ABC, abstractmethod
from builtins import RuntimeError
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaders, to_tree

import core


class ExplorerState:
    def __init__(self, indexes: list = None, linkage_matrix: list = None, clusters: np.ndarray | None = None, filters: list | None = None, level: int = 0):
        if (len(indexes) - 1) != len(linkage_matrix):
            raise ValueError("While building Explorer State. Linkage is not adequate for these indexes."
                             "\nIndexes: {}\nLinkage: {}".format(indexes, linkage_matrix))
        if clusters is not None and len(indexes) != len(clusters):
            raise ValueError("While building Explorer State. Indexes and clusters must have the same length."
                             "\nIndexes: {}\n Clusters: {}".format(indexes, clusters))
        if filters is not None and len(indexes) != len(filters):
            raise ValueError("While building Explorer State. Indexes and filters must have the same length."
                             "\nIndexes: {}\nClusters: {}".format(indexes, filters))
        self._indexes = np.array(indexes)
        self._linkage_matrix = np.array(linkage_matrix)
        self._clusters = None if clusters is None else np.array(clusters)
        self._filters = None if filters is None else np.array(filters)
        self._level = level

    def __copy__(self):
        return ExplorerState(self._indexes.copy(), self._linkage_matrix.copy(),
                             self._clusters.copy() if self._clusters is not None else None,
                             self._filters.copy() if self._filters is not None else None,
                             self._level)

    def __dict__(self) -> dict[str, Any]:
        return {
            "indexes": list(self._indexes),
            "linkage_matrix": [list(row) for row in self._linkage_matrix],
            "clusters": list(self._clusters) if self._clusters is not None else None,
            "filters": list(self._filters) if self._filters is not None else None,
            "level": self._level
        }

    @property
    def linkage_matrix(self):
        return self._linkage_matrix

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
    def __init__(self, state=None):
        self._state = state

    @property
    def state(self):
        return self._state

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
    def __init__(self, nclusters: int, state=None):
        super().__init__(state)
        if nclusters <= 1:
            raise ValueError(
                "Error while clustering. The number of clusters must be > 1. {} received.".format(nclusters))
        self._nclusters = nclusters

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        clusters = fcluster(state.linkage_matrix, self._nclusters, criterion='maxclust')
        return ExplorerState(state.indexes, state.linkage_matrix, clusters, level=state.level)

    def __dict__(self):
        return {
            'classname' : self.__class__.__name__,
            'kwargs' : {
                "state": None if self._state is None else self._state.__dict__(),
                "nclusters": self._nclusters
            }
        }


class ZoomInCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, selected_cluster: int, state=None):
        if selected_cluster <= 0:
            raise ValueError("While building ZoomInCommand. selected_cluster must be a positive integer ({} received)".format(selected_cluster))
        super().__init__(state)
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
        L, M = leaders(state.linkage_matrix, clusters)
        selected_root = L[np.where(M == self._selected_cluster)][0]
        tree = to_tree(state.linkage_matrix, rd=True)
        selected_tree = tree[1][selected_root]

        filtered_linkage = self.filter_linkage(state.linkage_matrix, selected_tree)

        leafs = selected_tree.pre_order(lambda x: x.id)
        filtered_indexes = np.array([state.indexes[x] for x in leafs])

        converted_ids = {leaf: i for i, leaf in enumerate(leafs)}
        converted_ids = converted_ids | {row[0]: (len(leafs) + i) for i, row in enumerate(filtered_linkage)}

        filtered_linkage_recalc_indexes = np.array(
            [[converted_ids[int(row[0])], converted_ids[int(row[1])], row[2], row[3]] for i, row in filtered_linkage])

        return ExplorerState(filtered_indexes, filtered_linkage_recalc_indexes, level=state.level+1)

    def __dict__(self):
        return {
            'classname': self.__class__.__name__,
            'kwargs': {
                "state": None if self._state is None else self._state.__dict__(),
                "selected_cluster": self._selected_cluster
            }
        }


class ApplyFiltersCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, keys: list, state=None):
        super().__init__(state)
        self._keys = np.array(keys)

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        if len(self._keys) != len(state.indexes):
            raise ValueError("ApplyFiltersCommand cannot be executed. Keys must be of same length as state's indexes.")
        return ExplorerState(state.indexes, state.linkage_matrix, state.clusters, self._keys, state.level)

    def __dict__(self):
        return {
            'classname': self.__class__.__name__,
            'kwargs': {
                "state": None if self._state is None else self._state.__dict__(),
                "keys": list(self._keys)
            }
        }


class CompositeCommand(ExplorerCommandDefaultPersistence):

    def __init__(self, command_list: list[ExplorerCommandDefaultPersistence], state=None):
        super().__init__(state=state)
        self._command_list = command_list

    def do(self, state: ExplorerState) -> ExplorerState:
        #self._state = copy.copy(state)
        actual_state = super().do(state)
        for command in self._command_list:
            actual_state = command.do(actual_state)
        return actual_state

    def __dict__(self):
        return {
            'classname': self.__class__.__name__,
            'kwargs': {
                "state": None if self._state is None else self._state.__dict__(),
                "command_list": [command.__dict__() for command in self._command_list]
            }
        }


class ParetoFrontExplorer:
    def __init__(self, state=None):
        self._actual_state = state
        self._done_commands = []
        self._undone_commands = []

    @property
    def actual_state(self) -> ExplorerState:
        return self._actual_state

    def load(self, store):
        if store is None:
            raise ValueError("While loading ParetoFrontExplorer. Store argument must be not None.")
        if not ("state" in store and "done_commands" in store and "undone_commands" in store):
            raise ValueError("While loading ParetoFrontExplorer. Store argument is inadequate because some key is lacking. \nstore received: {}".format(store))
        self._actual_state = ExplorerState(**store["state"])
        self._done_commands = []
        for command_as_dict in store["done_commands"]:
            command_as_dict["kwargs"]["state"] = None if command_as_dict["kwargs"]["state"] is None \
                else ExplorerState(**command_as_dict["kwargs"]["state"])
            command = eval("core."+command_as_dict["classname"])(**command_as_dict["kwargs"])
            self._done_commands.append(command)
        for command_as_dict in store["undone_commands"]:
            command_as_dict["kwargs"]["state"] = None if command_as_dict["kwargs"]["state"] is None \
                else ExplorerState(**command_as_dict["kwargs"]["state"])
            command = eval("core." + command_as_dict["classname"])(**command_as_dict["kwargs"])
            self._undone_commands.append(command)

    def save(self):
        return {
            "state": None if self._actual_state is None else self._actual_state.__dict__(),
            "done_commands": [command.__dict__() for command in self._done_commands],
            "undone_commands": [command.__dict__() for command in self._undone_commands]
        }

    def cluster(self, nclusters):
        pass

    def zoom_in(self, selected_cluster):
        pass

    def undo(self):
        if len(self._done_commands) == 0:
            raise RuntimeError("Explorer cannot undo because no command has executed yet.")
        last_command = self._done_commands.pop()
        self._actual_state = last_command.state
        self._undone_commands.append(last_command)

    def redo(self):
        raise NotImplementedError("Redo is not yet implemented")

    def restore(self):
        if self._done_commands:
            self._actual_state = self._done_commands[0].state
            self._done_commands.clear()
            self._undone_commands.clear()

    def apply_filter(self, filters):
        # Warning: stacked filter commands not allowed
        self._undone_commands.clear()
        filter_command = core.ApplyFiltersCommand(filters)
        self._actual_state = filter_command.do(self._actual_state)
        self._done_commands.append(filter_command)
