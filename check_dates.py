import pandas as pd
import numpy as np

def check_dates():
    print("=== 检查数据日期范围 ===")
    
    # 读取出租车数据
    print("\n1. 读取原始数据:")
    df_jan = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
    df_feb = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")
    df = pd.concat([df_jan, df_feb], ignore_index=True)
    
    # 转换时间戳
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['day'] = df['pickup_datetime'].dt.date
    
    # 只保留2023年1月和2月的数据
    df = df[df['pickup_datetime'].dt.year == 2023]
    df = df[df['pickup_datetime'].dt.month.isin([1, 2])]
    
    # 检查日期范围
    print("\n2. 检查日期范围:")
    print(f"数据开始日期: {df['day'].min()}")
    print(f"数据结束日期: {df['day'].max()}")
    print(f"总天数: {len(df['day'].unique())}")
    
    # 检查每天的记录数
    print("\n3. 每天的记录数:")
    daily_counts = df.groupby('day').size()
    print(daily_counts)
    
    # 检查是否有重复的日期
    print("\n4. 检查是否有重复的日期:")
    duplicate_days = daily_counts[daily_counts.index.duplicated()]
    if len(duplicate_days) > 0:
        print("发现重复的日期:")
        print(duplicate_days)
    else:
        print("没有重复的日期")

if __name__ == "__main__":
    check_dates() 