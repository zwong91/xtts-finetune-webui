import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset
import os
import json
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

class HFParquetProcessor:
    def __init__(self, dataset_name=None, local_path=None):
        """
        初始化处理器
        Args:
            dataset_name: Hugging Face数据集名称 (如 "mozilla-foundation/common_voice_11_0")
            local_path: 本地parquet文件路径
        """
        self.dataset_name = dataset_name
        self.local_path = local_path
        self.dataset = None
        
    def load_dataset_from_hf(self, split='train', streaming=False):
        """从Hugging Face加载数据集"""
        if not self.dataset_name:
            raise ValueError("请提供dataset_name")
        
        print(f"正在加载数据集: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name, split=split, streaming=streaming)
        print(f"数据集加载完成，包含 {len(self.dataset)} 条记录")
        return self.dataset
    
    def load_dataset_from_local(self):
        """从本地parquet文件加载数据集"""
        if not self.local_path:
            raise ValueError("请提供local_path")
        
        if os.path.isfile(self.local_path):
            # 单个parquet文件
            print(f"加载单个文件: {self.local_path}")
            df = pd.read_parquet(self.local_path)
            self.dataset = Dataset.from_pandas(df)
        elif os.path.isdir(self.local_path):
            # 文件夹包含多个parquet文件
            parquet_files = sorted(list(Path(self.local_path).glob("*.parquet")))
            if not parquet_files:
                raise ValueError(f"在 {self.local_path} 中未找到parquet文件")
            
            print(f"找到 {len(parquet_files)} 个parquet文件:")
            for file in parquet_files:
                print(f"  - {file.name}")
            
            dfs = []
            total_rows = 0
            for file in tqdm(parquet_files, desc="加载parquet文件"):
                df = pd.read_parquet(file)
                dfs.append(df)
                total_rows += len(df)
                print(f"  {file.name}: {len(df)} 条记录")
            
            print(f"合并数据集...")
            combined_df = pd.concat(dfs, ignore_index=True)
            self.dataset = Dataset.from_pandas(combined_df)
            print(f"总计: {total_rows} 条记录")
        
        print(f"本地数据集加载完成，包含 {len(self.dataset)} 条记录")
        return self.dataset
    
    def explore_dataset(self):
        """探索数据集结构"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        print("=== 数据集信息 ===")
        print(f"记录数量: {len(self.dataset)}")
        print(f"字段: {list(self.dataset.features.keys())}")
        
        print("\n=== 字段详细信息 ===")
        for key, feature in self.dataset.features.items():
            print(f"{key}: {feature}")
        
        print("\n=== 样本数据 ===")
        sample = self.dataset[0]
        for key, value in sample.items():
            if key == 'audio':
                if isinstance(value, dict):
                    print(f"{key}: {{")
                    for k, v in value.items():
                        if k == 'array':
                            print(f"  {k}: numpy数组，形状: {np.array(v).shape}")
                        else:
                            print(f"  {k}: {v}")
                    print("}")
                else:
                    print(f"{key}: {type(value)}")
            else:
                print(f"{key}: {value}")
        
        return self.dataset.features

    def save_organized_dataset(self, output_dir="./dataset_output", audio_format="wav", 
                             train_ratio=0.8, speaker_field=None, max_files=None):
        """
        保存组织好的数据集
        Args:
            output_dir: 输出目录
            audio_format: 音频格式 (wav, flac, mp3等)
            train_ratio: 训练集比例 (0.8表示80%训练，20%验证)
            speaker_field: 说话人字段名，如果为None则使用默认值
            max_files: 最大处理文件数量
        """
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        # 创建主输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据音频格式创建子文件夹
        audio_folder_name = f"{audio_format}s"  # wav -> wavs, flac -> flacs
        audio_dir = os.path.join(output_dir, audio_folder_name)
        os.makedirs(audio_dir, exist_ok=True)
        
        print(f"正在保存数据集到: {output_dir}")
        print(f"音频文件夹: {audio_folder_name}")
        print(f"音频格式: {audio_format}")
        
        # 准备数据列表
        dataset_records = []
        saved_count = 0
        max_count = max_files if max_files else len(self.dataset)
        
        print(f"开始处理 {min(max_count, len(self.dataset))} 条记录...")
        
        # 处理每个样本
        for i, sample in enumerate(tqdm(self.dataset, desc="处理数据")):
            if saved_count >= max_count:
                break
                
            # 获取文本
            text = sample.get('text', '').strip()
            if not text:
                continue
                
            # 获取说话人信息
            speaker_name = "Coqui"  # 默认说话人
            if speaker_field and speaker_field in sample:
                speaker_name = str(sample.get(speaker_field, "unknown"))
            elif 'speaker_id' in sample:
                speaker_name = str(sample.get('speaker_id', "unknown"))
            elif 'client_id' in sample:
                speaker_name = str(sample.get('client_id', "unknown"))
            elif 'speaker' in sample:
                speaker_name = str(sample.get('speaker', "unknown"))
            
            # 处理音频
            if 'audio' in sample and sample['audio'] is not None:
                audio_info = sample['audio']
                audio_saved = False
                
                try:
                    # 生成音频文件名
                    audio_filename = f"audio_{i:06d}_{saved_count:08d}.{audio_format}"
                    audio_filepath = os.path.join(audio_dir, audio_filename)
                    
                    if isinstance(audio_info, dict):
                        # 情况1: bytes格式
                        if 'bytes' in audio_info and audio_info['bytes'] is not None:
                            audio_bytes = audio_info['bytes']
                            
                            # 如果是WAV格式的bytes，直接保存
                            if audio_format == 'wav':
                                with open(audio_filepath, 'wb') as f:
                                    f.write(audio_bytes)
                                audio_saved = True
                            else:
                                # 需要转换格式，先保存为临时文件再转换
                                temp_path = audio_filepath + ".temp.wav"
                                with open(temp_path, 'wb') as f:
                                    f.write(audio_bytes)
                                
                                # 使用librosa读取并转换
                                try:
                                    y, sr = librosa.load(temp_path, sr=None)
                                    sf.write(audio_filepath, y, sr)
                                    os.remove(temp_path)
                                    audio_saved = True
                                except:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                        
                        # 情况2: 数组格式
                        elif 'array' in audio_info and audio_info['array'] is not None:
                            audio_array = np.array(audio_info['array'], dtype=np.float32)
                            sample_rate = audio_info.get('sampling_rate', 16000)
                            
                            if audio_array.ndim > 1:
                                audio_array = audio_array.flatten()
                            
                            sf.write(audio_filepath, audio_array, sample_rate)
                            audio_saved = True
                    
                    elif isinstance(audio_info, (list, np.ndarray)):
                        # 直接是数组
                        audio_array = np.array(audio_info, dtype=np.float32)
                        sample_rate = 16000  # 默认采样率
                        
                        if audio_array.ndim > 1:
                            audio_array = audio_array.flatten()
                        
                        sf.write(audio_filepath, audio_array, sample_rate)
                        audio_saved = True
                    
                    # 如果音频保存成功，添加到记录中
                    if audio_saved:
                        # 构建相对路径
                        relative_audio_path = f"{audio_folder_name}/{audio_filename}"
                        
                        record = {
                            'audio_file': relative_audio_path,
                            'text': text,
                            'speaker_name': speaker_name
                        }
                        dataset_records.append(record)
                        saved_count += 1
                        
                        if saved_count == 1:
                            print(f"✅ 成功保存第一个文件: {audio_filename}")
                            print(f"  文本: {text[:50]}...")
                            print(f"  说话人: {speaker_name}")
                
                except Exception as e:
                    if saved_count < 5:  # 只显示前5个错误
                        print(f"❌ 处理样本 {i} 时出错: {e}")
                    continue
        
        print(f"成功处理 {saved_count} 条记录")
        
        if saved_count == 0:
            print("⚠️  警告: 没有成功保存任何数据")
            return
        
        # 划分训练集和验证集
        if train_ratio < 1.0:
            train_records, eval_records = train_test_split(
                dataset_records, 
                train_size=train_ratio, 
                random_state=42,
                shuffle=True
            )
            
            print(f"数据集划分:")
            print(f"  训练集: {len(train_records)} 条")
            print(f"  验证集: {len(eval_records)} 条")
        else:
            # 全部作为训练集
            train_records = dataset_records
            eval_records = []
            print(f"全部数据作为训练集: {len(train_records)} 条")
        
        # 保存metadata CSV文件
        def save_metadata_csv(records, filename):
            if not records:
                return
                
            csv_path = os.path.join(output_dir, filename)
            with open(csv_path, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("audio_file|text|speaker_name\n")
                
                # 写入数据
                for record in records:
                    # 处理文本中的特殊字符
                    text = record['text'].replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
                    speaker = record['speaker_name'].replace('|', '\\|')
                    
                    f.write(f"{record['audio_file']}|{text}|{speaker}\n")
            
            print(f"已保存: {csv_path} ({len(records)} 条记录)")
        
        # 保存训练集metadata
        if train_records:
            save_metadata_csv(train_records, "metadata_train.csv")
        
        # 保存验证集metadata
        if eval_records:
            save_metadata_csv(eval_records, "metadata_eval.csv")
        
        # 生成数据集信息文件
        info = {
            "dataset_info": {
                "total_samples": saved_count,
                "train_samples": len(train_records),
                "eval_samples": len(eval_records),
                "audio_format": audio_format,
                "audio_folder": audio_folder_name,
                "train_ratio": train_ratio
            },
            "sample_record": dataset_records[0] if dataset_records else None
        }
        
        info_path = os.path.join(output_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 数据集处理完成！")
        print(f"输出目录: {output_dir}")
        print(f"  ├── {audio_folder_name}/          # 音频文件目录")
        print(f"  ├── metadata_train.csv    # 训练集元数据")
        if eval_records:
            print(f"  ├── metadata_eval.csv     # 验证集元数据")
        print(f"  └── dataset_info.json     # 数据集信息")
        
        return {
            'total_samples': saved_count,
            'train_samples': len(train_records),
            'eval_samples': len(eval_records),
            'output_dir': output_dir
        }

    def debug_audio_structure(self, num_samples=5):
        """调试音频数据结构"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        print("=== 音频数据结构调试 ===")
        print(f"数据集总数: {len(self.dataset)}")
        print(f"检查前 {num_samples} 个样本...")
        
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            print(f"\n--- 样本 {i} ---")
            print(f"所有字段: {list(sample.keys())}")
            
            # 检查每个字段
            for key, value in sample.items():
                if key == 'audio':
                    print(f"🎵 {key}:")
                    print(f"  类型: {type(value)}")
                    print(f"  值: {value}")
                    print(f"  是否为None: {value is None}")
                    
                    if value is not None:
                        if isinstance(value, dict):
                            print(f"  字典键: {list(value.keys())}")
                            for k, v in value.items():
                                if k == 'array':
                                    if v is not None:
                                        arr = np.array(v) if hasattr(np, 'array') else v
                                        print(f"    {k}: 类型={type(v)}, 长度={len(v) if hasattr(v, '__len__') else 'N/A'}")
                                        if hasattr(arr, 'shape'):
                                            print(f"         形状={arr.shape}, 数据类型={arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")
                                        if hasattr(v, '__len__') and len(v) > 0:
                                            print(f"         前几个值: {v[:5] if len(v) >= 5 else v}")
                                    else:
                                        print(f"    {k}: None")
                                else:
                                    print(f"    {k}: {v}")
                        elif isinstance(value, (list, tuple)):
                            print(f"  列表/元组长度: {len(value)}")
                            if len(value) > 0:
                                print(f"  前几个值: {value[:5]}")
                        elif hasattr(value, '__len__'):
                            print(f"  数组长度: {len(value)}")
                else:
                    # 其他字段简单显示
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {value_str}")

# 使用示例
def main():
    # 处理本地多个parquet文件
    processor = HFParquetProcessor(local_path="zh-taiwan/train")
    processor.load_dataset_from_local()
    
    # 探索数据集结构
    print("=" * 50)
    processor.explore_dataset()
    
    # 检查前几个样本的结构
    print("=" * 50)
    processor.debug_audio_structure(num_samples=3)
    
    # 保存组织好的数据集
    print("=" * 50)
    print("开始保存组织后的数据集...")
    
    result = processor.save_organized_dataset(
        output_dir="finetune_models/dataset",
        audio_format="wav",  # 可以改为 flac, mp3 等
        train_ratio=0.8,     # 80% 训练，20% 验证
        speaker_field="vvvv",  # 如果有特定的说话人字段名，在这里指定
        max_files=None       # 限制处理数量，None表示处理全部
    )
    
    if result:
        print(f"\n处理结果:")
        print(f"  总样本数: {result['total_samples']}")
        print(f"  训练样本: {result['train_samples']}")
        print(f"  验证样本: {result['eval_samples']}")
        print(f"  输出目录: {result['output_dir']}")

# 快速查看数据集信息的函数
def quick_inspect(parquet_dir):
    """快速查看parquet文件信息"""
    print(f"检查目录: {parquet_dir}")
    
    if not os.path.exists(parquet_dir):
        print(f"错误: 目录 {parquet_dir} 不存在")
        return
    
    parquet_files = sorted(list(Path(parquet_dir).glob("*.parquet")))
    if not parquet_files:
        print(f"在 {parquet_dir} 中未找到parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件:")
    
    total_rows = 0
    for file in parquet_files:
        # 读取parquet文件信息
        table = pq.read_table(file)
        rows = len(table)
        total_rows += rows
        
        print(f"  📁 {file.name}")
        print(f"     记录数: {rows:,}")
        print(f"     列数: {len(table.column_names)}")
        print(f"     列名: {table.column_names}")
        print(f"     文件大小: {file.stat().st_size / 1024 / 1024:.2f} MB")
        print()
    
    print(f"总记录数: {total_rows:,}")
    
    # 显示第一条记录示例
    if parquet_files:
        print("=" * 30)
        print("第一条记录示例:")
        first_table = pq.read_table(parquet_files[0])
        first_row = first_table.slice(0, 1).to_pandas().iloc[0]
        
        for col, value in first_row.items():
            if col == 'audio':
                print(f"  {col}: <音频数据>")
            else:
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"  {col}: {value_str}")


# uv pip install datasets pandas pyarrow librosa soundfile tqdm
if __name__ == "__main__":
    # 先快速查看数据集信息  
    quick_inspect("zh-taiwan/train")
    
    print("\n" + "=" * 80 + "\n")
    
    # 再进行完整处理
    main()