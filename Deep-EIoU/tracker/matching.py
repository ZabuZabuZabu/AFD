import numpy as np
import scipy
import lap
import networkx as nx
import csv
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(formation_matrix, indices, thresh):
    matched_cost = formation_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(formation_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(formation_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(formation_matrix, thresh):
    if formation_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(formation_matrix.shape[0])), tuple(range(formation_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(formation_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def expand(tlbr, e):
    
    t,l,b,r = tlbr
    w = r-l
    h = b-t
    expand_w = 2*w*e + w
    expand_h = 2*h*e + h

    new_tlbr = [t-expand_h//2,l-expand_w//2,b+expand_h//2,r+expand_w//2]

    return new_tlbr

def eious(atlbrs, btlbrs, e):
    """
    Compute cost based on EIoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    eious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if eious.size == 0:
        return eious

    atlbrs = np.array([expand(tlbr, e) for tlbr in atlbrs])
    btlbrs = np.array([expand(tlbr, e) for tlbr in btlbrs])

    eious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return eious


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype formation_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    formation_matrix = 1 - _ious

    return formation_matrix

def kalman_eiou_distance(atracks, btracks, expand):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype formation_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = eious(atlbrs, btlbrs, expand)
    formation_matrix = 1 - _ious

    return formation_matrix

def eiou_distance(atracks, btracks, expand):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype formation_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    _ious = eious(atlbrs, btlbrs, expand)
    formation_matrix = 1 - _ious

    return formation_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype formation_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    formation_matrix = 1 - _ious

    return formation_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: formation_matrix np.ndarray
    """

    formation_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if formation_matrix.size == 0:
        return formation_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    formation_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return formation_matrix

def gate_cost_matrix(kf, formation_matrix, tracks, detections, only_position=False):
    if formation_matrix.size == 0:
        return formation_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        formation_matrix[row, gating_distance > gating_threshold] = np.inf
    return formation_matrix


def fuse_motion(kf, formation_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if formation_matrix.size == 0:
        return formation_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        formation_matrix[row, gating_distance > gating_threshold] = np.inf
        formation_matrix[row] = lambda_ * formation_matrix[row] + (1 - lambda_) * gating_distance
    return formation_matrix


def fuse_iou(formation_matrix, tracks, detections):
    if formation_matrix.size == 0:
        return formation_matrix
    reid_sim = 1 - formation_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(formation_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(formation_matrix, detections):
    if formation_matrix.size == 0:
        return formation_matrix
    iou_sim = 1 - formation_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(formation_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

#ADD用
def displacement_distance(atracks, btracks):
    """
    Compute Displacement between each BBOX
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype formation_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
        
    # 各BBOXの中心座標を計算
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # # 結果の確認
    # print("Aの中心座標:", acenter_coords)
    # print("Bの中心座標:", bcenter_coords)
    
    # 各物体間の距離を計算
    acenter_distances=[]
    bcenter_distances=[]

    # atracksリスト内の距離計算
    for i, (x_i, y_i) in enumerate(acenter_coords):
        distances = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i == j:  # 自分自身との距離は0
                dist = 0
            else:
                dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            distances.append(dist)
        acenter_distances.append(distances)
    
    # atracksリスト内の距離計算
    for i, (x_i, y_i) in enumerate(bcenter_coords):
        distances = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i == j:  # 自分自身との距離は0
                dist = 0
            else:
                dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            distances.append(dist)
        bcenter_distances.append(distances)
    
    # # 結果の確認
    # print("Aの各BBOX中心座標の距離:", acenter_distances)
    # print("Bの各BBOX中心座標の距離:", bcenter_distances)
    
    rows, cols = len(acenter_distances),len(bcenter_distances)
    difference_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # ステップ 1: bcenter_distances の最初の行を選択
        b_distances = bcenter_distances[b_row]
        
        # ステップ 2: 値が小さい上位3つの要素を選択（ただし0を除く）
        top_b_distances = sorted([dist for dist in b_distances if dist > 0])[:3]
    
        for a_row in range(rows):
            # ステップ 3: acenter_distances の行を選択
            a_distances = acenter_distances[a_row]
            
            # ステップ 4: `top_b_distances` の各要素と `a_distances` の0でない要素を比較
            top_a_distances = sorted([dist for dist in a_distances if dist > 0])
            
            selected_a_distances = []
            for b_dist in top_b_distances:
                # 最も近い値を探して、重複なしで追加
                closest_a_dist = min((a_dist for a_dist in top_a_distances if a_dist not in selected_a_distances),
                                        key=lambda x: abs(x - b_dist), default=None)
                if closest_a_dist is not None:
                    selected_a_distances.append(closest_a_dist)
    
            # ステップ 5: top_b_distances と selected_a_distances の和の差を計算
            sum_b = sum(top_b_distances)
            sum_a = sum(selected_a_distances)
            difference_matrix[a_row, b_row] = abs(sum_b - sum_a)
            
        # ステップ 6: 次の行の処理に移る
    
    # # ステップ 7: 最終的に `difference_matrix` には各BBOXの差分が保存される
    # print("位置関係の差分行列:", difference_matrix)

    return difference_matrix

def displacement_distance_w_edge(atracks, btracks, margin):
    """
    Compute Displacement between each BBOX, excluding BBOXes close to screen edges
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype formation_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        # return x < margin or x > (screen_width - margin) or y < margin or y > (screen_height - margin) #データセット以外使うときはこっち＋他のコードも修正必要あり
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)#データセットはwidth1280,height720
    

    # Calculate intra-track distances
    acenter_distances = []
    bcenter_distances = []

    for i, (x_i, y_i) in enumerate(acenter_coords):
        distances = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i == j:
                dist = 0
            else:
                dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            distances.append(dist)
        acenter_distances.append(distances)
    
    for i, (x_i, y_i) in enumerate(bcenter_coords):
        distances = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i == j:
                dist = 0
            else:
                dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            distances.append(dist)
        bcenter_distances.append(distances)

    rows, cols = len(acenter_distances), len(bcenter_distances)
    difference_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        b_distances = bcenter_distances[b_row]
        
        # Filter out distances to BBOXes near screen edges
        top_b_distances = sorted(
            [dist for i, dist in enumerate(b_distances) if dist > 0 and not is_near_edge(bcenter_coords[i])]
        )[:3]
    
        for a_row in range(rows):
            a_distances = acenter_distances[a_row]
            
            # ステップ 4: `top_b_distances` の各要素と `a_distances` の0でない要素を比較
            top_a_distances = sorted([dist for dist in a_distances if dist > 0])#t-1はエッジ処理しない
            
            selected_a_distances = []
            for b_dist in top_b_distances:
                closest_a_dist = min(
                    (a_dist for a_dist in top_a_distances if a_dist not in selected_a_distances),
                    key=lambda x: abs(x - b_dist),
                    default=None
                )
                if closest_a_dist is not None:
                    selected_a_distances.append(closest_a_dist)
    
            sum_b = sum(top_b_distances)
            sum_a = sum(selected_a_distances)
            difference_matrix[a_row, b_row] = abs(sum_b - sum_a)

    return difference_matrix


def position_distance(atracks, btracks, margin):
    """
    Compute Displacement Vector between each BBOX, excluding BBOXes close to screen edges
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype formation_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)

    # Calculate intra-track vectors
    acenter_vectors = []
    bcenter_vectors = []

    for i, (x_i, y_i) in enumerate(acenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))  # ベクトルの計算
            else:
                vectors.append((0,0))#自分自身とのベクトルは０ベクトル
        acenter_vectors.append(vectors)
    
    for i, (x_i, y_i) in enumerate(bcenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))
            else: 
                vectors.append((0,0))
        bcenter_vectors.append(vectors)

    rows, cols = len(acenter_vectors), len(bcenter_vectors)
    position_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # 上位3つのベクトルを選択（b_vectors_top3）
        b_vectors_top3 = sorted(
            [v for i, v in enumerate(bcenter_vectors[b_row]) 
                if np.hypot(v[0], v[1]) > 0 and not is_near_edge(bcenter_coords[i])],
            key=lambda v: np.hypot(v[0], v[1])
        )[:3]

        sum_b_vector = np.sum(b_vectors_top3, axis=0) if b_vectors_top3 else np.array([0, 0])
        norm_b = np.linalg.norm(sum_b_vector)

        for a_row in range(rows):
            # `acenter_vectors[a_row]` から選択するためのフラグ
            used_indices = set()  # 選択済みのベクトルインデックス
            a_vectors_similar = []

            # b_vectors_top3 の各ベクトルについて、最も近い a_vector を選択
            for b_vec in b_vectors_top3:
                min_distance = float('inf')
                closest_a_vec = None
                closest_index = -1

                # 重複を避けながら最も近い a_vector を探索
                for idx, a_vec in enumerate(acenter_vectors[a_row]):
                    if idx in used_indices or np.hypot(a_vec[0], a_vec[1]) == 0:
                        continue  # 使用済みのベクトルやゼロベクトルはスキップ

                    distance = np.linalg.norm(np.array(a_vec) - np.array(b_vec))
                    if distance < min_distance:
                        min_distance = distance
                        closest_a_vec = a_vec
                        closest_index = idx

                # 最も近いベクトルをリストに追加し、インデックスを記録
                if closest_a_vec is not None:
                    a_vectors_similar.append(closest_a_vec)
                    used_indices.add(closest_index)

            # a_vectors_similar の合計を計算
            sum_a_vector = np.sum(a_vectors_similar, axis=0) if a_vectors_similar else np.array([0, 0])
            norm_a = np.linalg.norm(sum_a_vector)

            # コサイン類似度を計算
            if norm_b != 0 and norm_a != 0:
                cosine_similarity = np.dot(sum_b_vector, sum_a_vector) / (norm_b * norm_a)
            else:
                cosine_similarity = 0

            # 1 - コサイン類似度で方向の違いを示し、#大きさの差を加味
            magnitude_difference = abs(norm_b-norm_a) #同じ物体なら値が小さくなるはず
            position_matrix[a_row, b_row] = (1 - cosine_similarity) + 0.01*magnitude_difference

    return position_matrix


# Helper function to construct a graph from vectors
def build_graph(vectors, center_coord):
    """
    Build a graph based on given vectors and a single center coordinate.

    :param vectors: List of vectors to construct edges from.
    :param center_coord: Single center coordinate for all vectors.
    :return: A directed graph (networkx.DiGraph).
    """
    import networkx as nx
    import numpy as np

    G = nx.DiGraph()

    # Add the center coordinate as a node
    G.add_node("center", pos=center_coord)

    # Add vectors as edges with start and end points
    for dx, dy in vectors:
        if np.hypot(dx, dy) > 0:  # Ignore zero vectors
            start_coord = center_coord
            end_coord = (start_coord[0] + dx, start_coord[1] + dy)

            # Add end node with its position
            G.add_node(end_coord, pos=end_coord)

            # Add edge with additional attributes
            G.add_edge(
                "center",
                end_coord,
                length=np.hypot(dx, dy),  # Magnitude of the vector
                angle=np.arctan2(dy, dx)  # Angle of the vector
            )

    return G


# Helper function to compute Graph Edit Distance
def calculate_ged(graph1, graph2, mag1):
    #中心座標の差を計算しそれをコストとする関数
    def node_cost(n1, n2):
        pos1 = np.array(graph1.nodes[n1]['pos'])
        pos2 = np.array(graph2.nodes[n2]['pos'])
        return np.linalg.norm(pos1 - pos2)

    #
    def edge_cost(e1, e2):
        length1 = graph1.edges[e1]['length']
        length2 = graph2.edges[e2]['length']
        angle1 = graph1.edges[e1]['angle']
        angle2 = graph2.edges[e2]['angle']
        length_diff = abs(length1 - length2)
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # 周期性を考慮して最小の角度差を取る
        # return length_diff + mag* angle_diff
        return mag1*length_diff + angle_diff
        # return angle_diff

    total_cost = 0
    # for node1 in graph1.nodes:
    #     for node2 in graph2.nodes:
    #         total_cost += node_cost(node1, node2)

    # for edge1 in graph1.edges:
    #     for edge2 in graph2.edges:
    #         total_cost += edge_cost(edge1, edge2)

    for v1, v2 in zip(graph1.edges, graph2.edges):
        total_cost += edge_cost(v1, v2)

    # # Add penalties for unmatched nodes and edges
    # total_cost += len(graph1.nodes) + len(graph2.nodes)
    # total_cost += len(graph1.edges) + len(graph2.edges)

    return total_cost

def formation_distance(atracks, btracks, margin, mag, n_vec):
    """
    Compute Displacement Vector between each BBOX using Graph Edit Distance (GED).
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # Handle the case where there are no previous frame coordinates
    if len(atracks) == 0:
        #print("No previous frame data available. Skipping calculation.")
        return np.zeros((0, len(btracks))) # Return zero matrix

    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)

    # Calculate intra-track vectors
    acenter_vectors = []
    bcenter_vectors = []

    for i, (x_i, y_i) in enumerate(acenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))  # Calculate vector
            else:
                vectors.append((0, 0))  # Self-vector is zero
        acenter_vectors.append(vectors)

    for i, (x_i, y_i) in enumerate(bcenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))
            else:
                vectors.append((0, 0))
        bcenter_vectors.append(vectors)

    rows, cols = len(acenter_vectors), len(bcenter_vectors)
    position_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # Select top-3 vectors from bcenter_vectors
        b_vectors_top3 = sorted(
            [v for i, v in enumerate(bcenter_vectors[b_row]) 
             if np.hypot(v[0], v[1]) > 0 and not is_near_edge(bcenter_coords[i])],
            key=lambda v: np.hypot(v[0], v[1])
        )[:n_vec]

        # Build graph for b_vectors_top3
        graph_b = build_graph(b_vectors_top3, bcenter_coords[b_row])

        for a_row in range(rows):
            # `acenter_vectors[a_row]` から選択するためのフラグ
            used_indices = set()  # 選択済みのベクトルインデックス
            a_vectors_similar = []

            # b_vectors_top3 の各ベクトルについて、最も近い a_vector を選択
            for b_vec in b_vectors_top3:
                min_distance = float('inf')
                closest_a_vec = None
                closest_index = -1

                # 重複を避けながら最も近い a_vector を探索
                for idx, a_vec in enumerate(acenter_vectors[a_row]):
                    if idx in used_indices or np.hypot(a_vec[0], a_vec[1]) == 0:
                        continue  # 使用済みのベクトルやゼロベクトルはスキップ

                    distance = np.linalg.norm(np.array(a_vec) - np.array(b_vec))
                    if distance < min_distance:
                        min_distance = distance
                        closest_a_vec = a_vec
                        closest_index = idx

                # 最も近いベクトルをリストに追加し、インデックスを記録
                if closest_a_vec is not None:
                    a_vectors_similar.append(closest_a_vec)
                    used_indices.add(closest_index)
        
            # Build graph for a_vectors_similar
            graph_a = build_graph(a_vectors_similar, acenter_coords[a_row])

            # Calculate GED
            ged = calculate_ged(graph_a, graph_b, mag)

            position_matrix[a_row, b_row] = ged

    # # Normalize each column of position_matrix
    # for b_col in range(cols):
    #     column = position_matrix[:, b_col]
    #     min_value = np.min(column)
    #     max_value = np.max(column)

    #     if max_value != min_value:
    #         position_matrix[:, b_col] = (column - min_value) / (max_value - min_value)
    #     else:
    #         position_matrix[:, b_col] = 0  # All values in the column are the same

    return position_matrix

def formation_distance_2(atracks, btracks, margin, mag, n_vec):
    """
    Compute formation distance between each BBOX.
    search similar vectors with norm and angle.
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # Handle the case where there are no previous frame coordinates
    if len(atracks) == 0:
        #print("No previous frame data available. Skipping calculation.")
        return np.zeros((0, len(btracks))) # Return zero matrix

    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)

    # Calculate intra-track vectors
    acenter_vectors = []
    bcenter_vectors = []

    for i, (x_i, y_i) in enumerate(acenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))  # Calculate vector
            else:
                vectors.append((0, 0))  # Self-vector is zero
        acenter_vectors.append(vectors)

    for i, (x_i, y_i) in enumerate(bcenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))
            else:
                vectors.append((0, 0))
        bcenter_vectors.append(vectors)

    rows, cols = len(acenter_vectors), len(bcenter_vectors)
    formation_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # Select top-3 vectors from bcenter_vectors
        b_vectors_top3 = sorted(
            [v for i, v in enumerate(bcenter_vectors[b_row]) 
             if np.hypot(v[0], v[1]) > 0 and not is_near_edge(bcenter_coords[i])],
            key=lambda v: np.hypot(v[0], v[1])
        )[:n_vec]

        # Build graph for b_vectors_top3
        graph_b = build_graph(b_vectors_top3, bcenter_coords[b_row])

        for a_row in range(rows):
            # total_score = 0  # 各 a_row ごとにスコアをリセット
            # `acenter_vectors[a_row]` から選択するためのフラグ
            used_indices = set()  # 選択済みのベクトルインデックス
            a_vectors_similar = []

            # b_vectors_top3 の各ベクトルについて、最も近い a_vector を選択
            for b_vec in b_vectors_top3:
                min_score = float('inf')
                closest_a_vec = None
                closest_index = -1

                # 重複を避けながら最も近い a_vector を探索
                for idx, a_vec in enumerate(acenter_vectors[a_row]):
                    if idx in used_indices or np.hypot(a_vec[0], a_vec[1]) == 0:
                        continue  # 使用済みのベクトルやゼロベクトルはスキップ

                    # ノルム差を計算
                    norm_diff = abs(np.linalg.norm(a_vec) - np.linalg.norm(b_vec))

                    # 角度差を計算　シンプルな角度差
                    angle_a = np.arctan2(a_vec[1], a_vec[0])  # a_vec の角度（ラジアン）
                    angle_b = np.arctan2(b_vec[1], b_vec[0])  # b_vec の角度（ラジアン）
                    angle_diff = abs(angle_a - angle_b)  # 角度差（絶対値）

                    # 合計スコアを計算
                    score = mag*norm_diff + angle_diff  # ノルムの値が大きいので重みを少し小さくする．

                    # スコアが小さい場合に更新
                    if score < min_score:
                        min_score = score
                        closest_a_vec = a_vec
                        closest_index = idx

                # total_score += min_score

                # 最も近いベクトルをリストに追加し、インデックスを記録
                if closest_a_vec is not None:
                    a_vectors_similar.append(closest_a_vec)
                    used_indices.add(closest_index)
        
            # Build graph for a_vectors_similar
            graph_a = build_graph(a_vectors_similar, acenter_coords[a_row])

            # Calculate GED
            ged = calculate_ged(graph_a, graph_b, mag)

            formation_matrix[a_row, b_row] = ged

            # formation_matrix[a_row, b_row] = total_score

    # # Normalize each column of position_matrix
    # for b_col in range(cols):
    #     column = position_matrix[:, b_col]
    #     min_value = np.min(column)
    #     max_value = np.max(column)

    #     if max_value != min_value:
    #         position_matrix[:, b_col] = (column - min_value) / (max_value - min_value)
    #     else:
    #         position_matrix[:, b_col] = 0  # All values in the column are the same

    return formation_matrix


def formation_distance_record(atracks, btracks, margin, mag1, mag2, n_vec, frame_idx, enable_logging=False, log_path=None):
    """
    Compute formation distance between each BBOX.
    search similar vectors with norm and angle.
    record vectors

    Vector selection process:
    1. Select top n_vec vectors with similar magnitudes
    2. Among those vectors, select the one with the closest angle
    """
    # Initialize csv_log list only if logging is enabled
    csv_log = [] if enable_logging else None

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # formation_matrix = np.zeros((len(atracks), len(btracks)))
    # if formation_matrix.size == 0:
    #     return formation_matrix

    # Handle the case where there are no previous frame coordinates
    if len(atracks) == 0:
        return np.zeros((0, len(btracks))) # Return zero matrix
        # return np.zeros((0, 0)) # Return zero matrix
    
    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)

    # Calculate intra-track vectors
    acenter_vectors = []
    bcenter_vectors = []

    for i, (x_i, y_i) in enumerate(acenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))  # Calculate vector
            else:
                vectors.append((0, 0))  # Self-vector is zero
        acenter_vectors.append(vectors)

    for i, (x_i, y_i) in enumerate(bcenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))
            else:
                vectors.append((0, 0))
        bcenter_vectors.append(vectors)

    rows, cols = len(acenter_vectors), len(bcenter_vectors)
    formation_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # Create list of vectors with their indices
        vectors_with_indices = [
            (vec, i) for i, vec in enumerate(bcenter_vectors[b_row])
            if np.hypot(vec[0], vec[1]) > 0 and not is_near_edge(bcenter_coords[i])
        ]
        
        # Sort by vector magnitude while keeping track of original indices
        vectors_with_indices.sort(key=lambda x: np.hypot(x[0][0], x[0][1]))
        
        # Take top n_vec vectors and their indices
        b_vectors_top3 = []
        b_indices_top3 = []
        for vec, idx in vectors_with_indices[:n_vec]:
            b_vectors_top3.append(vec)
            b_indices_top3.append(idx)

        # # Build graph for b_vectors_top3
        # graph_b = build_graph(b_vectors_top3, bcenter_coords[b_row])

        for a_row in range(rows):
            total_score = 0  # 各 a_row ごとにスコアをリセット
            used_indices = set()
            a_vectors_similar = []
            a_indices_similar = []

            # Process each b_vector
            for b_vec in b_vectors_top3:
                min_score = float('inf')
                closest_a_vec = None
                closest_index = -1

                # Step 1: Find vectors with similar magnitude
                for idx, a_vec in enumerate(acenter_vectors[a_row]):
                    if idx in used_indices or np.hypot(a_vec[0], a_vec[1]) == 0:
                        continue

                    # ノルム差を計算
                    # norm_diff = abs(np.linalg.norm(a_vec) - np.linalg.norm(b_vec))

                    #ベクトルの差のノルムに変更（オリジナルのやり方）
                    norm_diff = np.linalg.norm(np.array(a_vec) - np.array(b_vec))

                    # 角度差を計算　シンプルな角度差
                    angle_a = np.arctan2(a_vec[1], a_vec[0])  # a_vec の角度（ラジアン）
                    angle_b = np.arctan2(b_vec[1], b_vec[0])  # b_vec の角度（ラジアン）
                    angle_diff = abs(angle_a - angle_b)  # 角度差（絶対値）
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # 周期性を考慮して最小の角度差を取る

                    # 合計スコアを計算
                    score = norm_diff + mag2*angle_diff 

                    # スコアが小さい場合に更新
                    if score < min_score:
                        min_score = score
                        closest_a_vec = a_vec
                        closest_index = idx

                total_score += min_score

                # Add the best matching vector
                if closest_a_vec is not None:
                    a_vectors_similar.append(closest_a_vec)
                    a_indices_similar.append(closest_index)
                    used_indices.add(closest_index)

            # CSV用のログを記録（ロギングが有効な場合のみ）
            if enable_logging:
                csv_log.append([
                    frame_idx,
                    b_row,
                    bcenter_coords[b_row],
                    b_vectors_top3,
                    b_indices_top3,
                    a_row,
                    acenter_coords[a_row],
                    a_vectors_similar,
                    a_indices_similar
                ])
        
            # # Build graph for a_vectors_similar
            # graph_a = build_graph(a_vectors_similar, acenter_coords[a_row])

            # # Calculate GED
            # ged = calculate_ged(graph_a, graph_b, mag1)

            # formation_matrix[a_row, b_row] = ged

            formation_matrix[a_row, b_row] = total_score

    # CSVへの書き込み（ロギングが有効な場合のみ）
    if enable_logging:
        # デフォルトのパスを設定
        default_path = "/home/zabu/Deep-EIoU/Deep-EIoU/YOLOX_outputs/yolox_x_ch_sportsmot/for_debag/AFD/check_vector/selected_vectors_log.csv"
        file_path = log_path if log_path is not None else default_path
        
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if frame_idx == 0:
                writer.writerow([
                    "Frame", "Graph_B_Index", "Graph_B_Center", "Graph_B_Vector", 
                    "Graph_B_Vector_Indices", "Graph_A_Index", "Graph_A_Center", 
                    "Graph_A_Vector", "Graph_A_Vector_Indices"
                ])
            writer.writerows(csv_log)

    return formation_matrix

def position_distance3(atracks, btracks, margin,mag):
    """
    Compute Displacement Vector between each BBOX using Graph Edit Distance (GED).
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # Handle the case where there are no previous frame coordinates
    if len(atracks) == 0:
        #print("No previous frame data available. Skipping calculation.")
        return np.zeros((0, len(btracks))) # Return zero matrix

    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)

    # Calculate intra-track vectors
    acenter_vectors = []
    bcenter_vectors = []

    for i, (x_i, y_i) in enumerate(acenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(acenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))  # Calculate vector
            else:
                vectors.append((0, 0))  # Self-vector is zero
        acenter_vectors.append(vectors)

    for i, (x_i, y_i) in enumerate(bcenter_coords):
        vectors = []
        for j, (x_j, y_j) in enumerate(bcenter_coords):
            if i != j:
                vectors.append((x_j - x_i, y_j - y_i))
            else:
                vectors.append((0, 0))
        bcenter_vectors.append(vectors)

    rows, cols = len(acenter_vectors), len(bcenter_vectors)
    position_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # Step 1: Define a reference angle θ
        reference_angle = -0.5*np.pi  # 任意の基準角度 (例: 0ラジアン)

        # Step 2: Select 3 vectors from bcenter_vectors closest to the reference angle
        b_vectors_top3 = sorted(
            [v for i, v in enumerate(bcenter_vectors[b_row]) 
                if np.hypot(v[0], v[1]) > 0 and not is_near_edge(bcenter_coords[i])],
            key=lambda v: abs(np.arctan2(v[1], v[0]) - reference_angle)
        )[:3]

        # Build graph for b_vectors_top3
        graph_b = build_graph(b_vectors_top3, bcenter_coords[b_row])

        for a_row in range(rows):
            used_indices = set()  # Step 3: Track used vectors from acenter_vectors
            a_vectors_similar = []

            for b_vec in b_vectors_top3:
                min_angle_diff = float('inf')
                closest_a_vec = None
                closest_index = -1

                # Step 3: Find the closest vector in angle to b_vec from acenter_vectors
                for idx, a_vec in enumerate(acenter_vectors[a_row]):
                    if idx in used_indices or np.hypot(a_vec[0], a_vec[1]) == 0:
                        continue

                    angle_b = np.arctan2(b_vec[1], b_vec[0])
                    angle_a = np.arctan2(a_vec[1], a_vec[0])
                    angle_diff = abs(angle_a - angle_b)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Adjust for circular angles　左回り，右回りで近い方

                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        closest_a_vec = a_vec
                        closest_index = idx

                if closest_a_vec is not None:
                    a_vectors_similar.append(closest_a_vec)
                    used_indices.add(closest_index)

            # Step 4: Build graph for a_vectors_similar
            graph_a = build_graph(a_vectors_similar, acenter_coords[a_row])

            # Calculate GED
            ged = calculate_ged(graph_a, graph_b, mag)

            # Store the similarity in the position matrix
            position_matrix[a_row, b_row] = ged

    return position_matrix

def formation_distance2(tracks, detections, edge_margin, mag, n_vec):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param edge_margin: float
    :param mag: float
    :return: formation_matrix np.ndarray
    """
    formation_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if formation_matrix.size == 0:
        return formation_matrix

    # Collect features and handle varying lengths
    det_features_list = [track.curr_feat for track in detections]
    track_features_list = [track.smooth_feat for track in tracks]

    # Determine maximum length of features
    max_det_length = max(len(f) for f in det_features_list)
    max_track_length = max(len(f) for f in track_features_list)
    feature_dim = len(det_features_list[0][0])  # Assumes all features have the same dimensionality

    # Pad detection features
    det_features = np.array([
        np.vstack((f, np.zeros((max_det_length - len(f), feature_dim)))) if len(f) < max_det_length else np.array(f)
        for f in det_features_list
    ])

    # Pad track features
    track_features = np.array([
        np.vstack((f, np.zeros((max_track_length - len(f), feature_dim)))) if len(f) < max_track_length else np.array(f)
        for f in track_features_list
    ])

    # # Debugging output
    # print("det_features shape:", det_features.shape)
    # print("track_features shape:", track_features.shape)

    # Calculate cost matrix
    formation_matrix = calculate_formation_distance(track_features, det_features, tracks, detections, edge_margin, mag, n_vec)
    return formation_matrix

def calculate_formation_distance(acenter_vectors,bcenter_vectors, atracks, btracks, margin, mag, n_vec):
    """
    Compute formation distance
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    
    # Calculate center coordinates for each BBOX
    acenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in atlbrs]
    bcenter_coords = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in btlbrs]

    # Handle the case where there are no previous frame coordinates
    if len(atracks) == 0:
        #print("No previous frame data available. Skipping calculation.")
        return np.zeros((0, len(btracks))) # Return zero matrix

    # Define function to check if a BBOX is near the screen edge
    def is_near_edge(coord):
        x, y = coord
        return x < margin or x > (1280 - margin) or y < margin or y > (720 - margin)

    rows, cols = len(acenter_vectors), len(bcenter_vectors)
    position_matrix = np.zeros((rows, cols))

    for b_row in range(cols):
        # Select top-3 vectors from bcenter_vectors
        b_vectors_top3 = sorted(
            [v for i, v in enumerate(bcenter_vectors[b_row]) 
             if i < len(bcenter_coords) and np.hypot(v[0], v[1]) > 0 and not is_near_edge(bcenter_coords[i])],
            key=lambda v: np.hypot(v[0], v[1])
        )[:n_vec]

        # Build graph for b_vectors_top3
        graph_b = build_graph(b_vectors_top3, bcenter_coords[b_row])

        for a_row in range(rows):
            # `acenter_vectors[a_row]` から選択するためのフラグ
            used_indices = set()  # 選択済みのベクトルインデックス
            a_vectors_similar = []

            # b_vectors_top3 の各ベクトルについて、最も近い a_vector を選択
            for b_vec in b_vectors_top3:
                min_distance = float('inf')
                closest_a_vec = None
                closest_index = -1

                # 重複を避けながら最も近い a_vector を探索
                for idx, a_vec in enumerate(acenter_vectors[a_row]):
                    if idx in used_indices or np.hypot(a_vec[0], a_vec[1]) == 0:
                        continue  # 使用済みのベクトルやゼロベクトルはスキップ

                    distance = np.linalg.norm(np.array(a_vec) - np.array(b_vec))
                    if distance < min_distance:
                        min_distance = distance
                        closest_a_vec = a_vec
                        closest_index = idx

                # 最も近いベクトルをリストに追加し、インデックスを記録
                if closest_a_vec is not None:
                    a_vectors_similar.append(closest_a_vec)
                    used_indices.add(closest_index)
        
            # Build graph for a_vectors_similar
            graph_a = build_graph(a_vectors_similar, acenter_coords[a_row])

            # Calculate GED
            ged = calculate_ged(graph_a, graph_b, mag)

            position_matrix[a_row, b_row] = ged

    return position_matrix

