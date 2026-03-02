import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np
import slangpy as spy

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """

        """
        這個 BVH constructor 做的事情是：
        從所有 primitives 開始，用 BFS 一層一層往下，
        在每個 node 用 SAH 找「最划算的切法」，
        原地重排 primitives，建立左右子節點，
        直到不能再切或節點數用完為止。
        """

        self.nodes: List[BVHNode] = []   # 用來裝最後的結果的

        # TODO: Student implementation starts here.

        self.primitives = primitives

        # build root
        root = BVHNode()

        # 說明這個 node 的 primitive 範圍是從 0 到 len(primitives)
        root.prim_left = 0
        root.prim_right = len(primitives)

        root.depth = 0

        # compute root bbox
        bbox = BoundingBox3D()
        for p in primitives:
            bbox = BoundingBox3D.union(bbox, p.bounding_box)
        root.bound = bbox

        self.nodes.append(root)

        # for BFS: (node_index, start, end, depth)
        # 從 root 開始，負責的範圍是 0 ~ len(primitives), depth = 0
        queue = [(0, 0, len(primitives), 0)]

        # BFS construction
        while queue and len(self.nodes) < max_nodes:  # 確定還有東西要處理，而且不能超過 max_nodes 數量上限
            node_index, start, end, depth = queue.pop(0)
            node = self.nodes[node_index]   # the node needs we are processing now
            prim_count = end - start        # the number of primitives

            # leaf condition
            if prim_count <= min_prim_per_node:
                continue

            parent_area = node.bound.area         # the surface area of the bbox of this node
            leaf_cost = parent_area * prim_count  # 打到這個 node 的機率 * 進來後要做多少 primitive test

            best_cost = float("inf")
            best_axis = None
            best_split_bucket = None

            # bucket SAH (check for all 3 axes)
            # 使用 bucket 近似所有可能的切法，並找最便宜的那個
            for axis in range(3):
                # the min and max on this axis
                axis_min = node.bound.min[axis]
                axis_max = node.bound.max[axis]
                extent = axis_max - axis_min

                if extent <= 1e-6:
                    continue

                B = num_thresholds  # the number of bucket
                buckets = [
                    {"count": 0, "bbox": BoundingBox3D()}   # (the number of primitives, union bbox)
                    for _ in range(B)
                ]

                # fill buckets, iterate through all the primitives in this node (from start to end)
                # 將所有 primitives 填入 bucket 中
                for i in range(start, end):
                    p = self.primitives[i]
                    c = p.bounding_box.center[axis]      # the center of this primitive on this axis
                    t = (c - axis_min) / extent          # map the center to the range of [0, 1]
                    b = min(B - 1, max(0, int(t * B)))   # transform a continuous real number t \in [0, 1] to a discrete bucket list of length B
                                   # float error # floor(t*B) 
                        # if int(t * B) == B -> go into the last bucket       
                        
                    buckets[b]["count"] += 1
                    buckets[b]["bbox"] = BoundingBox3D.union(
                        buckets[b]["bbox"], p.bounding_box  # union the bbox
                    )

                """
                # prefix / suffix accumulation

                Prefix / suffix accumulation 的目的，是讓你在 O(1) 時間內，
                知道「如果在第 s 個 bucket 切，左右兩邊各自的 bbox 與 primitive 數量是多少」。
                """
                left_bbox = [BoundingBox3D() for _ in range(B)]
                right_bbox = [BoundingBox3D() for _ in range(B)]
                left_count = [0] * B
                right_count = [0] * B

                # left prefix
                acc_bbox = BoundingBox3D()
                acc_count = 0
                for i in range(B):
                    acc_bbox = BoundingBox3D.union(acc_bbox, buckets[i]["bbox"])
                    acc_count += buckets[i]["count"]
                    left_bbox[i] = acc_bbox
                    left_count[i] = acc_count

                # right suffix
                acc_bbox = BoundingBox3D()
                acc_count = 0
                for i in reversed(range(B)):
                    acc_bbox = BoundingBox3D.union(acc_bbox, buckets[i]["bbox"])
                    acc_count += buckets[i]["count"]
                    right_bbox[i] = acc_bbox
                    right_count[i] = acc_count

                # evaluate split
                for s in range(1, B):
                    # if one side is empty -> invalid split -> continue
                    # s: 切下去的地方
                    if left_count[s - 1] == 0 or right_count[s] == 0:
                        continue

                    # Cost = AL​⋅NL ​+ AR​⋅NR​
                    cost = (
                        left_bbox[s - 1].area * left_count[s - 1]
                        + right_bbox[s].area * right_count[s]
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_split_bucket = s

            # SAH leaf test
            if best_axis is None or best_cost >= leaf_cost:  # 如果都大於 leaf_cost 了，那不如不做
                continue

            # 真的進行切割
            # recompute bucket index
            axis = best_axis
            axis_min = node.bound.min[axis]   # 取得這個軸向的範圍長度，之後需要在這個範圍內分割 bucket
            axis_max = node.bound.max[axis]
            extent = axis_max - axis_min
            B = num_thresholds
            split_bucket = best_split_bucket

            # test if primitive is in the left node
            def is_left(p: Primitive) -> bool:
                c = p.bounding_box.center[axis]
                t = (c - axis_min) / extent         # map the center to the range of [0, 1]
                b = min(B - 1, max(0, int(t * B)))  # transform a continuous real number t \in [0, 1] to a discrete bucket list of length B
                return b < split_bucket

            # in-place partition
            i = start
            j = end - 1
            while i <= j:
                if is_left(self.primitives[i]):
                    i += 1
                else:
                    # if primitive[i] belongs to the right, put it to right
                    # will check primitive[j] later, so we don't i += 1
                    self.primitives[i], self.primitives[j] = (
                        self.primitives[j],
                        self.primitives[i],
                    )
                    j -= 1

            # 分到最後，i 就會是中點
            mid = i
            # avoid empty subtree
            if mid == start or mid == end:
                continue

            # create children
            # cannot go over the limit of node number
            if len(self.nodes) + 2 > max_nodes:
                continue

            # left child
            left = BVHNode()
            left.prim_left = start
            left.prim_right = mid
            left.depth = depth + 1
            bbox = BoundingBox3D()
            for k in range(start, mid):
                bbox = BoundingBox3D.union(bbox, self.primitives[k].bounding_box)  # union of all the bbox
            left.bound = bbox

            left_index = len(self.nodes)
            self.nodes.append(left)
            node.left = left_index
            queue.append((left_index, start, mid, depth + 1))

            # right child
            right = BVHNode()
            right.prim_left = mid
            right.prim_right = end
            right.depth = depth + 1
            bbox = BoundingBox3D()
            for k in range(mid, end):
                bbox = BoundingBox3D.union(bbox, self.primitives[k].bounding_box)  # union of all the bbox
            right.bound = bbox

            right_index = len(self.nodes)   # this is the len(self.nodes) after buliding left child, as they are actually two different values
            self.nodes.append(right)
            node.right = right_index
            queue.append((right_index, mid, end, depth + 1))

            if on_progress:
                on_progress(len(self.nodes), max_nodes)

        # TODO: Student implementation ends here.


def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
