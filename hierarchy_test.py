import numpy as np
from scipy.cluster.hierarchy import complete, dendrogram, fcluster, leaders, to_tree, cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X = [[0, 0], [0, 1], [1, 0],
         [0, 4], [0, 3], [1, 4],
         [4, 0], [3, 0], [4, 1],
         [4, 4], [3, 4], [4, 3]]
    y = pdist(X)
    Z = complete(y)
    tree = to_tree(Z, rd=True)
    print("Linkage Matrix")
    print(Z)
    print("Tree")

    plt.figure()
    dendrogram(Z)

    clusters = fcluster(Z, 2, criterion='maxclust')
    L, M = leaders(Z, clusters)

    print(clusters)

    print("Leaders")
    print(L)
    print("Leaders cluster labels")
    print(M)

    slctd_cluster = 2
    print("Seleccionando cluster {}".format(slctd_cluster))

    #filtered_X = [i for i, x in enumerate(clusters) if x == slctd_cluster]
    slctd_root = L[np.where(M == slctd_cluster)][0]
    #print(slctd_root)

    slctd_tree = tree[1][slctd_root]
    complete_preorder = lambda node: [node.id] + (complete_preorder(node.left) if node.left else []) + (complete_preorder(node.right) if node.right else [])

    print(complete_preorder(slctd_tree))

    filter_linkage = lambda Z, node: [] if node.is_leaf() else filter_linkage(Z, node.left) + \
                                                               filter_linkage(Z, node.right) + \
                                                               [(node.id, list(Z[node.id - len(Z) - 1]))]
    filtered_linkage = filter_linkage(Z, slctd_tree)
    print(filtered_linkage)

    leafs = slctd_tree.pre_order(lambda x: x.id)
    filtered_X = [X[x] for x in leafs]

    print(leafs)
    print(filtered_X)
    converted_ids = {leaf:i for i, leaf in enumerate(leafs)}
    converted_ids = converted_ids | {row[0]: (len(leafs) + i) for i, row in enumerate(filtered_linkage)}
    print(converted_ids)

    filtered_linkage_recalc_indexes = np.array([[converted_ids[int(row[0])], converted_ids[int(row[1])], row[2], row[3]] for i, row in filtered_linkage])
    print(filtered_linkage_recalc_indexes)
    plt.show()

