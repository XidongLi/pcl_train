import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1


def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    """
    :param root:
    :param db: NxD
    :param db_sorted_idx_inv: NxD
    :param point_idx: M
    :param axis: scalar
    :param leaf_size: scalar
    :return:
    """
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5
        # === get the split position ===
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right,
                                           db,
                                           point_indices_sorted[middle_right_idx:],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)
    return root

def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)

    return False




