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
    
    def save_metadata(self, output_dir="./dataset_output"):
        """保存元数据到JSON文件"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = []
        print("正在提取元数据...")
        
        for i, sample in enumerate(tqdm(self.dataset)):
            record = {
                'id': i,
                'duration': sample.get('duration'),
                'text': sample.get('text'),
                'normalized_text': sample.get('normalized_text'),
            }
            
            # 如果有音频信息，添加音频元数据
            if 'audio' in sample and sample['audio'] is not None:
                audio_info = sample['audio']
                if isinstance(audio_info, dict):
                    record['audio_sample_rate'] = audio_info.get('sampling_rate')
                    if 'array' in audio_info:
                        record['audio_length'] = len(audio_info['array'])
            
            metadata.append(record)
        
        # 保存元数据
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"元数据已保存到: {metadata_file}")
        return metadata
    
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
        
        # 统计音频字段情况
        print(f"\n=== 音频字段统计 ===")
        audio_count = 0
        none_count = 0
        dict_count = 0
        array_count = 0
        
        for i, sample in enumerate(self.dataset):
            if 'audio' in sample:
                audio_count += 1
                if sample['audio'] is None:
                    none_count += 1
                elif isinstance(sample['audio'], dict):
                    dict_count += 1
                elif isinstance(sample['audio'], (list, tuple, np.ndarray)):
                    array_count += 1
        
        print(f"样本中:")
        print(f"  有audio字段: {audio_count}")
        print(f"  audio为None: {none_count}")
        print(f"  audio为字典: {dict_count}")
        print(f"  audio为数组: {array_count}")

    def save_audio_files(self, output_dir="./dataset_output", audio_format="wav", max_files=None):
        """保存音频文件"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        count = 0
        max_count = max_files if max_files else len(self.dataset)
        
        print(f"正在保存音频文件 (格式: {audio_format})...")
        print(f"音频保存路径: {audio_dir}")
        
        # 保存音频文件
        for i, sample in enumerate(tqdm(self.dataset, desc="保存音频")):
            if count >= max_count:
                break
                
            if 'audio' in sample and sample['audio'] is not None:
                audio_info = sample['audio']
                
                try:
                    # 处理不同的音频数据格式
                    audio_saved = False
                    
                    if isinstance(audio_info, dict):
                        # 情况1: bytes格式 - 完整的WAV文件数据
                        if 'bytes' in audio_info and audio_info['bytes'] is not None:
                            audio_bytes = audio_info['bytes']
                            
                            # 生成文件名
                            filename = f"audio_{i:06d}.wav"  # bytes通常是WAV格式
                            filepath = os.path.join(audio_dir, filename)
                            
                            # 直接保存二进制数据
                            with open(filepath, 'wb') as f:
                                f.write(audio_bytes)
                            
                            audio_saved = True
                            count += 1
                            
                            if count == 1:  # 第一个文件保存成功时显示信息
                                print(f"✅ 成功保存第一个文件: {filename}")
                                print(f"  文件大小: {len(audio_bytes)} bytes")
                                print(f"  文件格式: WAV (从bytes数据)")
                                
                                # 尝试读取音频信息
                                try:
                                    import wave
                                    with wave.open(filepath, 'rb') as wav_file:
                                        frames = wav_file.getnframes()
                                        sample_rate = wav_file.getframerate()
                                        duration = frames / sample_rate
                                        print(f"  采样率: {sample_rate} Hz")
                                        print(f"  时长: {duration:.2f} 秒")
                                except:
                                    print("  无法读取WAV文件详细信息")
                        
                        # 情况2: 数组格式 {'array': [...], 'sampling_rate': 16000}
                        elif 'array' in audio_info and audio_info['array'] is not None:
                            audio_array = np.array(audio_info['array'], dtype=np.float32)
                            sample_rate = audio_info.get('sampling_rate', 16000)
                            
                            # 生成文件名
                            filename = f"audio_{i:06d}.{audio_format}"
                            filepath = os.path.join(audio_dir, filename)
                            
                            # 确保音频数据是正确的格式
                            if audio_array.ndim > 1:
                                audio_array = audio_array.flatten()
                            
                            # 保存音频文件
                            sf.write(filepath, audio_array, sample_rate)
                            audio_saved = True
                            count += 1
                            
                            if count == 1:
                                print(f"✅ 成功保存第一个文件: {filename}")
                                print(f"  数组形状: {audio_array.shape}")
                                print(f"  采样率: {sample_rate}")
                                print(f"  时长: {len(audio_array)/sample_rate:.2f}秒")
                        
                        # 情况3: 路径格式 {'path': '...', 'sampling_rate': 16000}
                        elif 'path' in audio_info:
                            if count < 5:  # 只显示前5个路径信息
                                print(f"⚠️  样本 {i}: 音频为路径引用 {audio_info['path']}")
                            continue
                            
                    elif isinstance(audio_info, (list, np.ndarray)):
                        # 情况4: 直接是数组
                        audio_array = np.array(audio_info, dtype=np.float32)
                        sample_rate = 16000  # 默认采样率
                        
                        filename = f"audio_{i:06d}.{audio_format}"
                        filepath = os.path.join(audio_dir, filename)
                        
                        if audio_array.ndim > 1:
                            audio_array = audio_array.flatten()
                        
                        sf.write(filepath, audio_array, sample_rate)
                        audio_saved = True
                        count += 1
                    
                    if not audio_saved and i < 5:  # 只显示前5个的详细信息
                        print(f"❌ 跳过样本 {i}: 无法处理的音频格式")
                        
                except Exception as e:
                    if i < 5:  # 只显示前5个的错误信息
                        print(f"❌ 保存样本 {i} 时出错: {e}")
                    continue
        
        print(f"已保存 {count} 个音频文件到: {audio_dir}")
        if count == 0:
            print("⚠️  警告: 没有成功保存任何音频文件")
        else:
            print(f"🎉 成功保存了 {count} 个音频文件！")
        
        return count
    
    def save_text_files(self, output_dir="./dataset_output"):
        """保存文本文件"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        text_dir = os.path.join(output_dir, "texts")
        os.makedirs(text_dir, exist_ok=True)
        
        # 保存原始文本
        texts = []
        normalized_texts = []
        
        for sample in self.dataset:
            texts.append(sample.get('text', ''))
            normalized_texts.append(sample.get('normalized_text', ''))
        
        # 保存到文件
        with open(os.path.join(text_dir, "texts.txt"), 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        with open(os.path.join(text_dir, "normalized_texts.txt"), 'w', encoding='utf-8') as f:
            for text in normalized_texts:
                f.write(text + '\n')
        
        print(f"文本文件已保存到: {text_dir}")
        return len(texts)
    
    def export_to_csv(self, output_dir="./dataset_output"):
        """导出为CSV格式"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        data = []
        for i, sample in enumerate(self.dataset):
            record = {
                'id': i,
                'duration': sample.get('duration'),
                'text': sample.get('text'),
                'normalized_text': sample.get('normalized_text'),
            }
            
            # 添加音频信息
            if 'audio' in sample and sample['audio'] is not None:
                audio_info = sample['audio']
                if isinstance(audio_info, dict):
                    record['audio_sample_rate'] = audio_info.get('sampling_rate')
                    record['audio_filename'] = f"audio_{i:06d}.wav"
            
            data.append(record)
        
        # 保存CSV
        df = pd.DataFrame(data)
        csv_file = os.path.join(output_dir, "dataset.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"CSV文件已保存到: {csv_file}")
        return df

# 使用示例
def main():
    # 处理本地多个parquet文件
    processor = HFParquetProcessor(local_path="zh-taiwan/train")
    processor.load_dataset_from_local()
    
    # 探索数据集
    print("=" * 50)
    processor.explore_dataset()
    
    # 保存数据
    output_dir = "./zh_taiwan_dataset_output"
    
    print("=" * 50)
    print("开始保存数据...")
    
    # 保存元数据
    metadata = processor.save_metadata(output_dir)
    print(f"元数据包含 {len(metadata)} 条记录")
    
    # 保存音频文件 (如果数据量大，可以限制数量)
    audio_count = processor.save_audio_files(output_dir, max_files=len(metadata))
    print(f"保存了 {audio_count} 个音频文件")
    
    # 保存文本文件
    text_count = processor.save_text_files(output_dir)
    print(f"保存了 {text_count} 条文本记录")
    
    # 导出CSV
    df = processor.export_to_csv(output_dir)
    print(f"CSV包含 {len(df)} 行数据")
    
    print("=" * 50)
    print("数据处理完成！")
    print(f"输出目录: {output_dir}")

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

if __name__ == "__main__":
    # 先快速查看数据集信息  
    quick_inspect("zh-taiwan/train")
    
    print("\n" + "=" * 80 + "\n")
    
    # 再进行完整处理
    main()