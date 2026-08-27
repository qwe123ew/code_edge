# -*- coding: utf-8 -*-
"""
步骤2: 边缘划分 —— 判定"哪些像素属于同一条边缘"。

思路:
  1. Zhang-Suen 细化: 把 2~3px 宽的 Canny 边缘细成 1px 骨架;
  2. 骨架上按 8-邻域个数分类:
       邻域数 == 1 -> 端点(边缘的起点/终点)
       邻域数 == 2 -> 内部点(Edge)
       邻域数 >= 3 -> 交叉点(多条边缘交汇处, V)
  3. 移除交叉点后, 剩余像素的每个 8-连通分量就是"同一条边缘"(一段);
     交叉点处的邻接关系被记录下来, 供步骤4建图(四色染色)用;
  4. 端点数 == 0 的段为闭合轮廓(候选物体边界) —— 体现"边缘与物体的关系"。

这正好对应图论中平面图的 V(交叉点) - E(边缘段) + F(面/物体) 结构。
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- 细化
def zhang_suen_thin(binary):
    """Zhang-Suen 并行细化算法(向量化实现), 反复迭代直到骨架稳定。"""
    b = binary.astype(bool).copy()
    while True:
        changed = False
        for step in (1, 2):
            P = np.pad(b, 1)
            p2 = P[0:-2, 1:-1]   # N
            p3 = P[0:-2, 2:]     # NE
            p4 = P[1:-1, 2:]     # E
            p5 = P[2:, 2:]       # SE
            p6 = P[2:, 1:-1]     # S
            p7 = P[2:, 0:-2]     # SW
            p8 = P[1:-1, 0:-2]   # W
            p9 = P[0:-2, 0:-2]   # NW
            u = lambda x: x.astype(np.uint8)
            B = u(p2) + u(p3) + u(p4) + u(p5) + u(p6) + u(p7) + u(p8) + u(p9)
            seq = [p2, p3, p4, p5, p6, p7, p8, p9]
            A = np.zeros_like(b, dtype=np.uint8)
            for i in range(8):
                A += ((~seq[i]) & seq[(i + 1) % 8]).astype(np.uint8)  # 0->1 转变数
            mark = b & (B >= 2) & (B <= 6) & (A == 1)
            if step == 1:
                mark = mark & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                mark = mark & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if mark.any():
                b &= ~mark
                changed = True
        if not changed:
            return b


# ---------------------------------------------------------------- 邻域统计
def neighbor_count(mask):
    """每个像素 8-邻域中前景个数"""
    k = np.ones((3, 3), np.int32)
    k[1, 1] = 0
    return ndi.convolve(mask.astype(np.int32), k, mode='constant', cval=0)


# ---------------------------------------------------------------- 划分
def partition_edges(edge_mask, min_len=4, endpoint_adj_r=2.5):
    """把边缘二值图划分为若干条"边缘段"。

    返回:
      lab     : int 图, 0=背景, 1..K 为各边缘段 id
      jmask   : 交叉点掩码(属于 >=3 段交汇处)
      adj     : dict {seg_id: set(相邻 seg_id)} —— 供四色染色建图
      meta    : dict(sizes, endpoints, is_closed, n_junctions, ...)
    """
    skel = zhang_suen_thin(edge_mask)
    nc = neighbor_count(skel)
    jmask = skel & (nc >= 3)          # 交叉点
    seg = skel & ~jmask

    lab, n = ndi.label(seg, structure=np.ones((3, 3), np.int32))

    # 丢弃过短的噪声段
    if n > 0:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep_ids = np.where(sizes >= min_len)[0]
        keep_ids = keep_ids[keep_ids > 0]
        keep = np.isin(lab, keep_ids)
    else:
        keep = np.zeros_like(seg)
        keep_ids = np.array([], int)
    lab = np.where(keep, lab, 0)

    # 紧凑重编号 1..K
    ids = np.unique(lab)
    ids = ids[ids > 0]
    remap = np.zeros(lab.max() + 1 if lab.max() > 0 else 1, np.int32)
    for new, old in enumerate(ids, start=1):
        remap[old] = new
    lab = remap[lab]
    K = len(ids)

    # 端点: 划分后在段内邻域数为 1 的像素(靠近交叉点或真正端点)
    nc_seg = neighbor_count(lab > 0)
    ep_mask = (lab > 0) & (nc_seg == 1)

    # 每段端点数(0 => 闭合轮廓)
    ep_count = np.zeros(K + 1, int)
    for sid in np.unique(lab[ep_mask]):
        ep_count[sid] = int(((lab == sid) & ep_mask).sum())

    sizes_new = np.zeros(K + 1, int)
    for sid in range(1, K + 1):
        sizes_new[sid] = int((lab == sid).sum())

    # ---- 邻接关系(四色染色的图结构) ----
    adj = {i: set() for i in range(1, K + 1)}

    # (a) 共享同一交叉点的段两两相邻
    Pl = np.pad(lab, 1)
    for (jy, jx) in np.argwhere(jmask):
        y, x = jy + 1, jx + 1  # pad 后坐标
        neigh = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = int(Pl[y + dy, x + dx])
                if v > 0:
                    neigh.add(v)
        neigh = sorted(neigh)
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                adj[neigh[a]].add(neigh[b])
                adj[neigh[b]].add(neigh[a])

    # (b) 端点距离很近(中间断了一个像素)的段视为相邻(可选的边缘重连)
    eps = np.argwhere(ep_mask)
    if len(eps) > 1:
        tree = cKDTree(eps)
        for i, j in tree.query_pairs(r=endpoint_adj_r):
            a, b = int(lab[eps[i, 0], eps[i, 1]]), int(lab[eps[j, 0], eps[j, 1]])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

    # ---- 近闭合检测: 两端距离 ≤ gap_threshold (默认 5px) ----
    # 见 edge_closure.compute_near_closed_segments 说明
    from edge_closure import compute_near_closed_segments
    near_closed, gap_distances = compute_near_closed_segments(
        lab, edge_mask, gap_threshold=5.0)

    meta = dict(
        K=K,
        sizes=sizes_new,
        endpoint_pixels=ep_count,
        is_closed=(ep_count == 0),     # 严格闭合: 无端点 (Canny 几乎为 0)
        near_closed=near_closed,       # 近闭合: 两端距离 ≤ 5px
        gap_distances=gap_distances,   # 近闭合段的端点距离
        n_junctions=int(jmask.sum()),
    )
    return lab, jmask, adj, meta, skel
