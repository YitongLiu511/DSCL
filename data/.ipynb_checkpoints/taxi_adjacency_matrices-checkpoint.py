# taxi_static_adjacency.py
import geopandas as gpd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist

def load_zones(shapefile_path):
    """加载地理区域数据（自动转换坐标系）"""
    zones = gpd.read_file(shapefile_path)
    return zones.to_crs(epsg=32618)  # 转换为纽约UTM投影

def compute_distance_matrix(zones):
    """地理距离矩阵（千米）"""
    coords = np.array([[geom.x, geom.y] for geom in zones.geometry.centroid])
    return cdist(coords, coords, metric='euclidean') / 1000

def compute_connectivity_matrix(zones):
    """地理邻接矩阵（拓扑连接）"""
    n = len(zones)
    conn = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if i != j and zones.iloc[i].geometry.touches(zones.iloc[j].geometry):
                conn[i, j] = 1
    return conn

def compute_poi_similarity(zones):
    """POI特征相似度矩阵"""
    encoder = OneHotEncoder()
    features = zones[['borough', 'zone']].astype(str).apply('_'.join, axis=1)
    encoded = encoder.fit_transform(features.values.reshape(-1,1))
    return cosine_similarity(encoded)

if __name__ == "__main__":
    try:
        # 加载数据（替换为实际路径）
        zones = load_zones("taxi_zones.shp")
        
        # 生成矩阵
        print("Generating distance matrix...")
        dist = compute_distance_matrix(zones)
        
        print("Generating connectivity matrix...")
        conn = compute_connectivity_matrix(zones)
        
        print("Generating POI similarity matrix...")
        poi = compute_poi_similarity(zones)
        
        # 保存结果
        np.savez_compressed("static_adjacency.npz",
                          distance=dist.astype(np.float32),
                          connectivity=conn,
                          poi_similarity=poi.astype(np.float32))
        print("矩阵已保存至 static_adjacency.npz")
        print("字段列表:", zones.columns.tolist())  # 应包含'borough'或'zone'字段
        print("区域数量:", zones.shape[0])        # 官方行政区应为5个，但交通/统计可能细分更多
    except Exception as e:
        print(f"错误: {str(e)}")