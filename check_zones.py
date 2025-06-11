import pandas as pd
import numpy as np
import geopandas as gpd

def check_zones():
    print("=== 检查区域数量关系 ===")
    
    # 1. 检查区域查找表
    print("\n1. 检查区域查找表 (taxi_zone_lookup.csv):")
    lookup = pd.read_csv("data/taxi _zone_lookup.csv")
    lookup_zones = sorted(lookup['LocationID'].unique())
    print(f"查找表中的区域数量: {len(lookup_zones)}")
    print(f"查找表中的区域ID范围: {min(lookup_zones)} 到 {max(lookup_zones)}")
    
    # 2. 检查Shape文件
    print("\n2. 检查Shape文件 (taxi_zones.shp):")
    zones = gpd.read_file('data/taxi_zones.shp')
    shape_zones = sorted(zones['LocationID'].unique())
    print(f"Shape文件中的区域数量: {len(shape_zones)}")
    print(f"Shape文件中的区域ID范围: {min(shape_zones)} 到 {max(shape_zones)}")
    
    # 3. 检查原始数据
    print("\n3. 检查原始数据 (yellow_tripdata):")
    df_jan = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
    df_feb = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")
    df = pd.concat([df_jan, df_feb], ignore_index=True)
    
    # 获取所有唯一的区域ID
    data_zones = sorted(set(df['PULocationID'].unique()) | set(df['DOLocationID'].unique()))
    print(f"原始数据中的区域数量: {len(data_zones)}")
    print(f"原始数据中的区域ID范围: {min(data_zones)} 到 {max(data_zones)}")
    
    # 4. 检查邻接矩阵
    print("\n4. 检查邻接矩阵 (static_adjacency.npz):")
    adj_data = np.load("data/static_adjacency.npz")
    adj = adj_data['connectivity']
    print(f"邻接矩阵中的区域数量: {adj.shape[0]}")
    print(f"邻接矩阵中的区域ID范围: 1 到 {adj.shape[0]}")
    
    # 5. 检查区域差异
    print("\n5. 检查区域差异:")
    
    # 在查找表中但不在Shape文件中的区域
    missing_in_shape = set(lookup_zones) - set(shape_zones)
    print(f"\n在查找表中但不在Shape文件中的区域: {sorted(missing_in_shape)}")
    if missing_in_shape:
        print("\n这些区域的详细信息:")
        print(lookup[lookup['LocationID'].isin(missing_in_shape)][['LocationID', 'Borough', 'Zone']])
    
    # 在原始数据中但不在Shape文件中的区域
    extra_in_data = set(data_zones) - set(shape_zones)
    print(f"\n在原始数据中但不在Shape文件中的区域: {sorted(extra_in_data)}")
    
    # 在Shape文件中但不在原始数据中的区域
    missing_in_data = set(shape_zones) - set(data_zones)
    print(f"\n在Shape文件中但不在原始数据中的区域: {sorted(missing_in_data)}")
    
    # 6. 总结
    print("\n6. 总结:")
    print(f"区域查找表中的区域数量: {len(lookup_zones)}")
    print(f"Shape文件中的区域数量: {len(shape_zones)}")
    print(f"原始数据中的区域数量: {len(data_zones)}")
    print(f"邻接矩阵中的区域数量: {adj.shape[0]}")
    print("\n最终使用的区域数量: 263 (来自邻接矩阵)")

if __name__ == "__main__":
    check_zones() 