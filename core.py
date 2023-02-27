# import copy
from abc import ABC, abstractmethod
from builtins import RuntimeError
from copy import copy
from typing import Any

import numpy as np
from numpy import int32
from scipy.cluster.hierarchy import fcluster, leaders, to_tree

import core


class ExplorerState:
    def __init__(self, indexes: list[int] = None, linkage_matrix: list = None, clusters: dict[int, int] | None = None,
                 filters: dict[int, bool] | None = None, level: int = 0):
        if (len(indexes) - 1) != len(linkage_matrix):
            raise ValueError("While building Explorer State. Linkage is not adequate for these indexes."
                             "\nIndexes: {}\nLinkage: {}".format(indexes, linkage_matrix))
        if clusters is not None and not all([idx in clusters.keys() for idx in indexes]):
            raise ValueError("While building Explorer State. All indexes must be mapped by clusters dict."
                             "\nIndexes: {}\n Clusters: {}".format(indexes, clusters))
        if filters is not None and not all([idx in filters.keys() for idx in indexes]):
            raise ValueError("While building Explorer State. All indexes must be mapped by filters dict."
                             "\nIndexes: {}\nFilters: {}".format(indexes, filters))
        self._indexes = np.array(indexes)
        self._linkage_matrix = np.array(linkage_matrix)
        self._clusters = clusters
        self._filters = filters
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
            "clusters": None if self._clusters is None
            else {str(idx): c for idx, c in self._clusters.items()},
            "filters": None if self._filters is None
            else {str(idx): tv for idx, tv in self._filters.items()},
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
        return self._indexes if self._filters is None \
            else self._indexes[[self._filters[idx] for idx in self._indexes]]

    @property
    def clusters(self):
        return self._clusters

    @property
    def filters(self):
        return self._filters

    @property
    def level(self):
        return self._level

    @clusters.setter
    def clusters(self, clusters_dict: dict[int, int]):
        if clusters_dict is not None and not all([idx in clusters_dict.keys() for idx in self._indexes]):
            raise ValueError("While building Explorer State. All indexes must be mapped by clusters dict."
                             "\nIndexes: {}\n Clusters: {}".format(self._indexes, clusters_dict))
        self._clusters = clusters_dict

    @filters.setter
    def filters(self, filters_dict: dict[int, bool]):
        if filters_dict is not None and not all([idx in filters_dict.keys() for idx in self._indexes]):
            raise ValueError("While building Explorer State. All indexes must be mapped by filters dict."
                             "\nIndexes: {}\nFilters: {}".format(self._indexes, filters_dict))
        self._filters = filters_dict


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
        # self._state = copy.copy(state)
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
        new_state = state.__copy__()
        filters = new_state.filters
        new_state.filters = None
        new_state.clusters = {new_state.indexes[i]: cluster_number for i, cluster_number in enumerate(clusters)}
        new_state.filters = filters
        #print("Clusters generated")
        #print(new_state.clusters)
        return new_state

    def __dict__(self):
        return {
            'classname': self.__class__.__name__,
            'kwargs': {
                "state": None if self._state is None else self._state.__dict__(),
                "nclusters": self._nclusters
            }
        }


class ZoomInCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, selected_cluster: int, state=None):
        if selected_cluster <= 0:
            raise ValueError(
                "While building ZoomInCommand. selected_cluster must be a positive integer ({} received)".format(
                    selected_cluster))
        super().__init__(state)
        self._selected_cluster = selected_cluster

    def filter_linkage(self, linkage, node):
        return [] if node.is_leaf() else self.filter_linkage(linkage, node.left) \
                                         + self.filter_linkage(linkage, node.right) \
                                         + [(node.id, list(linkage[node.id - len(linkage) - 1]))]

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        clusters = state.clusters
        # Must clear filters to properly make the zoom in
        new_state = state.__copy__()
        filters = new_state.filters
        new_state.filters = None

        # intermediate_state = copy(state)
        # intermediate_state.filters = None
        if clusters is None:
            raise ValueError("Zoom in command cannot be executed in an unclustered state.")
        if self._selected_cluster > (NCLUSTERS := max(state.clusters.values())):
            raise ValueError(
                "Zoom in command cannot be executed. Selected cluster is out of bounds. Number of clusters is {} and I want to select {}.".format(
                    NCLUSTERS, self._selected_cluster))
        clusters_array = np.array([int32(clusters[idx]) for idx in new_state.indexes]) # no me pregunten por qué pero no anda esto
        #clusters_array2 = fcluster(state.linkage_matrix, NCLUSTERS, criterion='maxclust')
        L, M = leaders(new_state.linkage_matrix, clusters_array)
        selected_root = L[np.where(M == self._selected_cluster)][0]
        tree = to_tree(new_state.linkage_matrix, rd=True)
        selected_tree = tree[1][selected_root]

        filtered_linkage = self.filter_linkage(new_state.linkage_matrix, selected_tree)

        leafs = selected_tree.pre_order(lambda x: x.id)
        filtered_indexes = [new_state.indexes[x] for x in leafs]

        converted_ids = {leaf: i for i, leaf in enumerate(leafs)}
        converted_ids = converted_ids | {row[0]: (len(leafs) + i) for i, row in enumerate(filtered_linkage)}

        filtered_linkage_recalc_indexes = np.array(
            [[converted_ids[int(row[0])], converted_ids[int(row[1])], row[2], row[3]] for i, row in filtered_linkage])

        return ExplorerState(filtered_indexes, filtered_linkage_recalc_indexes, filters=filters, level=new_state.level + 1)

    def __dict__(self):
        return {
            'classname': self.__class__.__name__,
            'kwargs': {
                "state": None if self._state is None else self._state.__dict__(),
                "selected_cluster": self._selected_cluster
            }
        }


class ApplyFiltersCommand(ExplorerCommandDefaultPersistence):
    def __init__(self, keys: dict[int, bool], state=None):
        super().__init__(state)
        self._keys = keys

    def do(self, state: ExplorerState) -> ExplorerState:
        super().do(state)
        # if len(self._keys) != len(state.indexes):
        #    raise ValueError("ApplyFiltersCommand cannot be executed. Keys must be of same length as state's indexes.")
        new_state = state.__copy__()
        new_state.filters = self._keys
        return new_state

    def __dict__(self):
        return {
            'classname': self.__class__.__name__,
            'kwargs': {
                "state": None if self._state is None else self._state.__dict__(),
                "keys": {str(idx): tv for idx, tv in self._keys.items()}
            }
        }


class CompositeCommand(ExplorerCommandDefaultPersistence):

    def __init__(self, command_list: list[ExplorerCommandDefaultPersistence], state=None):
        super().__init__(state=state)
        self._command_list = command_list

    def do(self, state: ExplorerState) -> ExplorerState:
        # self._state = copy.copy(state)
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

    def _build_state(self, state_as_dict):
        if "filters" in state_as_dict and state_as_dict["filters"] is not None:
            state_as_dict["filters"] = {int(idx): tf for idx, tf in state_as_dict["filters"].items()}
        if "clusters" in state_as_dict and state_as_dict["clusters"] is not None:
            state_as_dict["clusters"] = {int(idx): c for idx, c in state_as_dict["clusters"].items()}
        state = ExplorerState(**state_as_dict)
        return state

    def _build_command(self, command_as_dict):
        command_as_dict["kwargs"]["state"] = None if command_as_dict["kwargs"]["state"] is None \
                else self._build_state(command_as_dict["kwargs"]["state"])
        if "keys" in command_as_dict["kwargs"] and command_as_dict["kwargs"]["keys"] is not None:
            command_as_dict["kwargs"]["keys"] = {int(idx): tv for idx, tv in command_as_dict["kwargs"]["keys"].items()}
        command = eval("core." + command_as_dict["classname"])(**command_as_dict["kwargs"])
        return command

    def load(self, store):
        if store is None:
            raise ValueError("While loading ParetoFrontExplorer. Store argument must be not None.")
        if not ("state" in store and "done_commands" in store and "undone_commands" in store):
            raise ValueError(
                "While loading ParetoFrontExplorer. Store argument is inadequate because some key is lacking. \nstore received: {}".format(
                    store))
        self._actual_state = self._build_state(store["state"])

        self._done_commands = []
        self._done_commands = [self._build_command(command_as_dict) for command_as_dict in store["done_commands"]]
        #for command_as_dict in store["done_commands"]:
        #    command_as_dict["kwargs"]["state"] = None if command_as_dict["kwargs"]["state"] is None \
        #        else ExplorerState(**command_as_dict["kwargs"]["state"])
        #    command = eval("core." + command_as_dict["classname"])(**command_as_dict["kwargs"])
        #    self._done_commands.append(command)

        self._undone_commands = [self._build_command(command_as_dict) for command_as_dict in store["undone_commands"]]
        #for command_as_dict in store["undone_commands"]:
        #    command_as_dict["kwargs"]["state"] = None if command_as_dict["kwargs"]["state"] is None \
        #        else ExplorerState(**command_as_dict["kwargs"]["state"])
        #    command = eval("core." + command_as_dict["classname"])(**command_as_dict["kwargs"])
        #    self._undone_commands.append(command)

    def save(self):
        return {
            "state": None if self._actual_state is None else self._actual_state.__dict__(),
            "done_commands": [command.__dict__() for command in self._done_commands],
            "undone_commands": [command.__dict__() for command in self._undone_commands]
        }

    def cluster(self, nclusters):
        self._undone_commands.clear()

        try:
            cluster_command = core.ClusterCommand(nclusters)
            self._actual_state = cluster_command.do(self._actual_state)
            self._done_commands.append(cluster_command)

        except:
            print("Algo pasó con el comando cluster")

    def zoom_in(self, selected_cluster):
        self._undone_commands.clear()

        try:
            zoomin_command = core.ZoomInCommand(selected_cluster)
            self._actual_state = zoomin_command.do(self._actual_state)
            self._done_commands.append(zoomin_command)

        except:
            print("Algo pasó con el comando zoom in")

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
        try:
            filter_command = core.ApplyFiltersCommand(filters)
            self._actual_state = filter_command.do(self._actual_state)
            self._done_commands.append(filter_command)
        except:
            print("Algo pasó con el comando filter")
