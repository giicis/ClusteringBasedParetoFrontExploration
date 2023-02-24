import json
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, is_valid_linkage
from scipy.spatial.distance import pdist, squareform

import core
from util import distance


class TestCore(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_json("../assets/pareto_front.json", orient='index', dtype={'reqs': str, 'stks': str})
        max_cost = max(cls.data["cost"])
        max_profit = max(cls.data["profit"])
        cls.requirements = pd.read_json("../assets/requirements.json")
        cls.stakeholders = pd.read_json("../assets/stakeholders.json")
        distance_matrix = squareform(pdist(cls.data, lambda x, y: distance(x, y, max_profit, max_cost)))
        cls.linkage = linkage(distance_matrix, method="complete")

    def test_build_state(self):
        # self.assertIsNotNone(core.ExplorerState(self.data["id"], self.linkage))        # este también funciona
        #print(self.data["id"])
        #print(np.array((self.data["id"])))
        self.assertIsNotNone(core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage))

    def test_undo_not_done_command(self):
        self.assertRaises(RuntimeError, core.ZoomInCommand(selected_cluster=2).undo)

    def test_cluster(self):
        self.assertIsNotNone(core.ClusterCommand(4).do(core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)))

    def test_zoomin_do(self):
        # Arrange
        nclusters = 4
        selected_cluster = 3
        original_state = core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)
        cluster = core.ClusterCommand(nclusters)
        zoomin = core.ZoomInCommand(selected_cluster=selected_cluster)

        # Act
        clustered_state = cluster.do(original_state)
        new_state = zoomin.do(clustered_state)

        # Assert
        self.assertIsNone(new_state.clusters)
        self.assertTrue(is_valid_linkage(new_state.linkage_matrix))
        self.assertTrue(original_state.level == new_state.level-1)

    def test_zoomin_do_and_undo(self):
        # Arrange
        nclusters = 4
        selected_cluster = 3
        original_state = core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)
        cluster = core.ClusterCommand(nclusters)
        zoomin = core.ZoomInCommand(selected_cluster=selected_cluster)

        # Act
        clustered_state = cluster.do(original_state)
        zoomin.do(clustered_state)
        undone_state = zoomin.undo()

        # Assert
        self.assertTrue(all([us_i in original_state.indexes for us_i in undone_state.indexes]))
        self.assertTrue(is_valid_linkage(undone_state.linkage_matrix))

    def test_filter(self):
        # Arrange
        original_state = core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)
        filter_command = core.ApplyFiltersCommand(self.data.index < 100)

        # Act
        filtered_state = filter_command.do(original_state)

        # Assert
        self.assertTrue(len(original_state.indexes) >= len(filtered_state.indexes))

    def test_persistence(self):
        original_state = core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)
        print(original_state.__dict__())

    def test_sava_load_with_kwargs(self):
        original_state = core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)
        d = original_state.__dict__()
        new_state = core.ExplorerState(**d)
        print(new_state)

    def test_build_with_class_name(self):
        nclusters = 4
        original_state = core.ExplorerState(indexes=self.data.index, linkage_matrix=self.linkage)
        name = core.ClusterCommand.__class__.__name__
        cluster = eval("core.ClusterCommand")(nclusters)

        cluster_dict = cluster.__dict__()
        print(cluster_dict)

        cluster2 = eval("core."+cluster_dict["classname"])(**cluster_dict["kwargs"])
        cluster2.do(original_state)