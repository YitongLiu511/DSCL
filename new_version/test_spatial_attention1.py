import torch
import numpy as np
import os
from spatial_attention1 import process_dynamic_stream_data

def test_spatial_attention1():
    print("=== 测试 spatial_attention1.py 动态流处理 ===")
    
    try:
        # 1. 检查时间注意力处理结果是否存在
        temporal_cache = 'data/processed/temporal_attention_processed.npy'
        if not os.path.exists(temporal_cache):
            print(f"错误：未找到时间注意力处理结果 {temporal_cache}")
            print("请先运行 test_process_normal.py 生成时间注意力特征")
            return
        
        # 2. 加载时间注意力处理后的数据
        print("\n1. 加载时间注意力处理后的数据...")
        temporal_output = np.load(temporal_cache)
        print(f"时间注意力输出形状: {temporal_output.shape}")
        
        # 3. 检查数据形状是否符合预期
        if temporal_output.shape != (263, 2016, 2):
            print(f"警告：数据形状 {temporal_output.shape} 与预期 (263, 2016, 2) 不符")
            print("但会继续尝试处理...")
        
        # 4. 调用动态流处理
        print("\n2. 开始动态流处理...")
        dynamic_scores, attention_weights = process_dynamic_stream_data(
            temporal_output,
            d_model=256,
            n_heads=8,
            device='cpu'
        )
        
        # 5. 验证输出
        print("\n3. 验证输出结果...")
        print(f"动态流分数形状: {dynamic_scores.shape}")
        print(f"注意力权重形状: {attention_weights.shape}")
        
        # 6. 检查是否成功保存
        dynamic_scores_path = 'data/processed/dynamic_scores.npy'
        if os.path.exists(dynamic_scores_path):
            print(f"✅ 动态流分数已成功保存到: {dynamic_scores_path}")
            
            # 验证保存的文件
            saved_scores = np.load(dynamic_scores_path)
            print(f"保存的文件形状: {saved_scores.shape}")
            print(f"保存的文件与返回值一致: {np.array_equal(dynamic_scores.detach().cpu().numpy(), saved_scores)}")
        else:
            print("❌ 动态流分数保存失败")
        
        # 7. 检查与静态流分数的兼容性
        static_scores_path = 'data/processed/static_scores.npy'
        if os.path.exists(static_scores_path):
            print(f"\n4. 检查与静态流分数的兼容性...")
            static_scores = np.load(static_scores_path)
            print(f"静态流分数形状: {static_scores.shape}")
            
            if dynamic_scores.shape == static_scores.shape:
                print("✅ 动态流和静态流分数维度匹配，可以进行双流对比学习")
            else:
                print("❌ 动态流和静态流分数维度不匹配")
                print(f"  动态流: {dynamic_scores.shape}")
                print(f"  静态流: {static_scores.shape}")
        else:
            print(f"\n4. 未找到静态流分数 {static_scores_path}")
            print("请先运行 test_process_normal.py 生成静态流分数")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {str(e)}")
        import traceback
        print("错误详情:")
        print(traceback.format_exc())

def main():
    """
    主函数，用于加载时间流数据，通过动态空间流处理，并保存动态分数和特征输出。
    """
    print("--- 开始处理动态空间流 ---")
    
    # --- 1. 定义文件路径 ---
    temporal_output_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/temporal_attention_processed.npy'
    dynamic_scores_save_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/dynamic_scores.npy'
    dynamic_output_save_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/dynamic_stream_output.npy' # 新增输出路径
    
    # --- 2. 加载时间流输出数据 ---
    try:
        temporal_output = np.load(temporal_output_path)
        print(f"时间注意力输出形状: {temporal_output.shape}")
        
        # 3. 检查数据形状是否符合预期
        if temporal_output.shape != (263, 2016, 2):
            print(f"警告：数据形状 {temporal_output.shape} 与预期 (263, 2016, 2) 不符")
            print("但会继续尝试处理...")
        
        # 4. 运行动态空间流处理 ---
        print("4. 正在通过动态空间流处理...")
        # process_dynamic_stream_data 返回 (dynamic_scores, attention_output)
        dynamic_scores, attention_output = process_dynamic_stream_data(
            torch.from_numpy(temporal_output).float(), 
            d_model=256, 
            n_heads=8
        )
        print(f"   - 成功生成动态分数，形状: {dynamic_scores.shape}")
        print(f"   - 成功生成特征输出，形状: {attention_output.shape}")

        # --- 5. 保存动态分数和特征输出 ---
        print("5. 正在保存结果...")
        np.save(dynamic_scores_save_path, dynamic_scores.detach().numpy())
        print(f"   - 动态分数已保存到: {dynamic_scores_save_path}")
        
        np.save(dynamic_output_save_path, attention_output.detach().numpy()) # 保存特征输出
        print(f"   - 特征输出已保存到: {dynamic_output_save_path}")

        print("\n--- 处理完成 ---")

    except Exception as e:
        print(f"\n❌ 测试过程中出错: {str(e)}")
        import traceback
        print("错误详情:")
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 