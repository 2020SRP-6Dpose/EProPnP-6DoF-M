"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import pickle as pkl
from plyfile import PlyData, PlyElement
import numpy as np

# ======================================
# .ply file
# ======================================
def load_ply_vtx(pth):
    """
    load object vertices
    :param pth: str
    :return: pts: (N, 3)
    """
    ply = PlyData.read(pth)
    vtx = ply['vertex']
    pts = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    return pts

def load_ply_vtx_expand(pth, expand_ratio=0):
    """
    load and expand object vertices via interpolation（插值） on edges 
    :param pth: str
    :param expand_ratio: int, expanding ratio, should be >= 0, default: 0, i.e. w/o expanding.
    :return: pts: (N, 3)
    """
    f = open(pth, 'r')  # read模式打开pth处文件
    assert f.readline().strip() == "ply"
    while True:
        line = f.readline().strip()
        if line.startswith('element vertex'):
            N = int(line.split()[-1])
        if line.startswith('element face'):
            F = int(line.split()[-1])
        if line == 'end_header':
            break
    print('loading vertices...')
    pts = []
    for _ in tqdm(range(N)):
        pts.append(np.float32(f.readline().split()[:3]))
    if expand_ratio > 0:
        expand_ratio = float(expand_ratio+1)
        inter_num = int(expand_ratio)
        print('loading expanded vertices...')
        pts_expand = []
        for _ in tqdm(range(F)):
            f_vtx_num, *f_vtx_idx = f.readline().strip().split()
            for i in range(int(f_vtx_num)):
                for j in range(int(f_vtx_num)-1):
                    vtx_i = pts[int(f_vtx_idx[i])]
                    vtx_j = pts[int(f_vtx_idx[j])]
                    step = 1 / expand_ratio * (vtx_j-vtx_i)
                    for k in range(inter_num):
                        pts_expand.append(vtx_i + (k+1) * step)
        pts = pts+pts_expand
    f.close()
    return np.array(pts)

def write_ply(save_pth, pts, text=True):
    """
    save point cloud (vtx only) into a .ply file
    :param save_path: str, path to save, for example, '/yy/XX.ply'
    :param pts: array-like, size=(N,3), point_cloud
    """ 
    # pts = pts.tolist()
    pts = [(pts[i,0], pts[i,1], pts[i,2]) for i in range(pts.shape[0])]
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')]) # float4
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_pth)


# ======================================
# .pkl file
# ======================================
def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pkl.load(f)

def save_pickle(data, pkl_path):
    os.system("mkdir -p {}".format(os.path.dirname(pkl_path)))
    with open(pkl_path, "wb") as f:
        pkl.dump(data, f)