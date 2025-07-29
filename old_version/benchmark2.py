#!/usr/bin/env python3
"""
Enhanced Deep Learning Benchmark Tool - 增强版深度学习基准测试工具
支持多种模型类型和数据集的交互式基准测试

主要功能：
1. 交互式设备选择 (CPU/GPU)
2. 模型类型选择 (Detection/Classification)
3. 数据集选择 (MNIST/CIFAR-10/COCO/ImageNet)
4. 详细性能统计和CSV输出
5. 结果可视化

Author: Enhanced for professional AI benchmarking
Platform: Ubuntu 22.04 + NVIDIA GPU support
"""

import argparse
import json
import csv
import os
import sys
import time
import socket
import threading
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Deep Learning Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm

# System Monitoring
import psutil
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")

# Object Detection (if available)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

class InteractiveBenchmark:
    def __init__(self):
        self.device = None
        self.model_type = None
        self.model = None
        self.dataset_name = None
        self.dataloader = None
        self.results = []
        
        # Monitoring
        self.monitoring = True
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_memory = deque(maxlen=1000)
        self.gpu_utilization = deque(maxlen=1000)
        
        self.total_samples = 0
        self.start_time = None
        
        # Available models
        self.detection_models = {
            '1': {'name': 'YOLOv8n', 'model': 'yolov8n.pt', 'type': 'yolo'},
            '2': {'name': 'YOLOv8s', 'model': 'yolov8s.pt', 'type': 'yolo'},
            '3': {'name': 'YOLOv8m', 'model': 'yolov8m.pt', 'type': 'yolo'},
        }
        
        self.classification_models = {
            '1': {'name': 'ResNet18', 'model': 'resnet18', 'type': 'timm'},
            '2': {'name': 'ResNet50', 'model': 'resnet50', 'type': 'timm'},
            '3': {'name': 'EfficientNet-B0', 'model': 'efficientnet_b0', 'type': 'timm'},
            '4': {'name': 'EfficientNet-B3', 'model': 'efficientnet_b3', 'type': 'timm'},
            '5': {'name': 'Vision Transformer', 'model': 'vit_base_patch16_224', 'type': 'timm'},
            '6': {'name': 'MobileNet-V3', 'model': 'mobilenetv3_large_100', 'type': 'timm'},
        }

    def interactive_setup(self):
        """交互式设置"""
        print("="*60)
        print("深度学习模型基准测试工具")
        print("="*60)
        
        # 1. 设备选择
        self.select_device()
        
        # 2. 模型类型选择
        self.select_model_type()
        
        # 3. 具体模型选择
        self.select_model()
        
        # 4. 数据集选择
        self.select_dataset()
        
        print("\n" + "="*60)
        print("配置总结：")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.model_type}")
        print(f"数据集: {self.dataset_name}")
        print("="*60)
        
        confirm = input("\n确认开始测试? (y/n): ").lower().strip()
        if confirm != 'y':
            print("测试已取消")
            sys.exit(0)

    def select_device(self):
        """选择计算设备"""
        print("\n1. 选择计算设备:")
        print("1) CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"2) CUDA:{i} - {torch.cuda.get_device_name(i)}")
        else:
            print("   (CUDA不可用)")
        
        while True:
            choice = input("请选择设备 (输入数字): ").strip()
            if choice == '1':
                self.device = 'cpu'
                break
            elif choice == '2' and torch.cuda.is_available():
                self.device = 'cuda:0'
                break
            else:
                print("无效选择，请重新输入")
        
        print(f"已选择设备: {self.device}")

    def select_model_type(self):
        """选择模型类型"""
        print("\n2. 选择模型类型:")
        print("1) 图像分类 (Classification)")
        if ULTRALYTICS_AVAILABLE:
            print("2) 目标检测 (Object Detection)")
        else:
            print("2) 目标检测 (需要安装 ultralytics)")
        
        while True:
            choice = input("请选择模型类型 (输入数字): ").strip()
            if choice == '1':
                self.model_type = 'classification'
                break
            elif choice == '2' and ULTRALYTICS_AVAILABLE:
                self.model_type = 'detection'
                break
            elif choice == '2':
                print("目标检测需要安装 ultralytics: pip install ultralytics")
            else:
                print("无效选择，请重新输入")
        
        print(f"已选择模型类型: {self.model_type}")

    def select_model(self):
        """选择具体模型"""
        print(f"\n3. 选择{self.model_type}模型:")
        
        if self.model_type == 'classification':
            for key, value in self.classification_models.items():
                print(f"{key}) {value['name']}")
            
            while True:
                choice = input("请选择模型 (输入数字): ").strip()
                if choice in self.classification_models:
                    selected = self.classification_models[choice]
                    self.model_info = selected
                    print(f"已选择模型: {selected['name']}")
                    break
                else:
                    print("无效选择，请重新输入")
                    
        elif self.model_type == 'detection':
            for key, value in self.detection_models.items():
                print(f"{key}) {value['name']}")
            
            while True:
                choice = input("请选择模型 (输入数字): ").strip()
                if choice in self.detection_models:
                    selected = self.detection_models[choice]
                    self.model_info = selected
                    print(f"已选择模型: {selected['name']}")
                    break
                else:
                    print("无效选择，请重新输入")

    def select_dataset(self):
        """选择数据集"""
        print("\n4. 选择数据集:")
        
        if self.model_type == 'classification':
            print("1) MNIST (手写数字, 28x28)")
            print("2) CIFAR-10 (小物体分类, 32x32)")
            print("3) ImageNet 验证集样本 (224x224, 需要下载)")
            
            datasets = {
                '1': {'name': 'MNIST', 'func': self.load_mnist},
                '2': {'name': 'CIFAR-10', 'func': self.load_cifar10},
                '3': {'name': 'ImageNet-Sample', 'func': self.load_imagenet_sample}
            }
            
        elif self.model_type == 'detection':
            print("1) COCO 验证集样本 (需要下载)")
            print("2) 预设测试图像")
            
            datasets = {
                '1': {'name': 'COCO-Sample', 'func': self.load_coco_sample},
                '2': {'name': 'Test-Images', 'func': self.load_test_images}
            }
        
        while True:
            choice = input("请选择数据集 (输入数字): ").strip()
            if choice in datasets:
                selected = datasets[choice]
                self.dataset_name = selected['name']
                self.load_dataset_func = selected['func']
                print(f"已选择数据集: {selected['name']}")
                break
            else:
                print("无效选择，请重新输入")

    def load_model(self):
        """加载选定的模型"""
        print(f"\n正在加载模型: {self.model_info['name']}...")
        
        try:
            if self.model_type == 'classification':
                if self.model_info['type'] == 'timm':
                    self.model = timm.create_model(
                        self.model_info['model'], 
                        pretrained=True,
                        num_classes=1000
                    )
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    
            elif self.model_type == 'detection':
                if self.model_info['type'] == 'yolo':
                    self.model = YOLO(self.model_info['model'])
                    
            print("模型加载成功!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    def load_mnist(self):
        """加载MNIST数据集"""
        print("正在加载MNIST数据集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        self.dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        print(f"MNIST数据集加载完成，共{len(dataset)}个样本")

    def load_cifar10(self):
        """加载CIFAR-10数据集"""
        print("正在加载CIFAR-10数据集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        self.dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        print(f"CIFAR-10数据集加载完成，共{len(dataset)}个样本")

    def load_imagenet_sample(self):
        """加载ImageNet样本数据集"""
        print("ImageNet样本数据集设置...")
        print("请下载ImageNet验证集到 './data/imagenet/val' 目录")
        print("下载地址: https://www.image-net.org/download.php")
        print("或使用少量样本图像进行测试")
        
        # 创建简单的测试数据
        self.create_synthetic_dataset(224, 1000)

    def load_coco_sample(self):
        """加载COCO样本数据集"""
        print("COCO样本数据集设置...")
        print("请下载COCO 2017验证集到 './data/coco' 目录")
        print("下载地址: http://cocodataset.org/#download")
        
        # 创建简单的测试数据
        self.create_synthetic_detection_data()

    def load_test_images(self):
        """加载预设测试图像"""
        print("使用预设测试图像...")
        self.create_synthetic_detection_data()

    def create_synthetic_dataset(self, img_size, num_classes):
        """创建合成数据集用于测试"""
        print(f"创建合成数据集 ({img_size}x{img_size}, {num_classes}类)...")
        
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000, img_size=224, num_classes=1000):
                self.size = size
                self.img_size = img_size
                self.num_classes = num_classes
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 生成随机图像
                img = torch.randn(3, self.img_size, self.img_size)
                label = torch.randint(0, self.num_classes, (1,)).item()
                return img, label
        
        dataset = SyntheticDataset(1000, img_size, num_classes)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        print("合成数据集创建完成")

    def create_synthetic_detection_data(self):
        """创建合成检测数据"""
        print("创建合成检测数据...")
        
        # 生成一些测试图像路径
        self.test_images = []
        for i in range(100):
            # 创建随机图像并保存
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = f"./temp_test_img_{i}.jpg"
            
            # 这里应该保存图像，但为了简化我们只保存路径
            self.test_images.append(img_path)
        
        print(f"创建{len(self.test_images)}个测试图像")

    def monitor_resources(self):
        """监控系统资源"""
        if torch.cuda.is_available() and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                handle = None
        else:
            handle = None
        
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU和内存
                self.cpu_usage.append(process.cpu_percent(interval=0.1))
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU监控
                if torch.cuda.is_available():
                    # GPU内存
                    gpu_mem = torch.cuda.memory_allocated(0)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory
                    self.gpu_memory.append((gpu_mem / gpu_total) * 100)
                    
                    # GPU利用率
                    if handle:
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            self.gpu_utilization.append(util.gpu)
                        except:
                            self.gpu_utilization.append(0)
                
                time.sleep(0.1)
                
            except Exception as e:
                break

    def run_classification_benchmark(self):
        """运行分类模型基准测试"""
        print("\n开始分类模型基准测试...")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.dataloader):
                batch_start = time.time()
                
                # 预处理时间
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # 推理时间
                inf_start = time.time()
                output = self.model(data)
                inf_time = (time.time() - inf_start) * 1000  # ms
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # 记录时间
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # 进度显示
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * len(data)} samples...")
                
                # 限制测试样本数
                if batch_idx >= 100:  # 限制测试批次
                    break
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'batch_times': batch_times
        }

    def run_detection_benchmark(self):
        """运行检测模型基准测试"""
        print("\n开始检测模型基准测试...")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        
        # 使用合成图像进行测试
        for i in range(min(100, len(self.test_images))):
            # 创建随机图像进行测试
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 执行推理
            results = self.model(img, device=self.device, verbose=False)
            
            # 获取时间信息
            if hasattr(results[0], 'speed'):
                speed = results[0].speed
                preprocessing_times.append(speed.get('preprocess', 0))
                inference_times.append(speed.get('inference', 0))
                postprocessing_times.append(speed.get('postprocess', 0))
            
            self.total_samples += 1
            
            if i % 10 == 0:
                print(f"Processed {i} images...")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times
        }

    def run_benchmark(self):
        """运行基准测试"""
        print("\n开始基准测试...")
        
        # 启动资源监控
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.start_time = time.time()
        
        try:
            if self.model_type == 'classification':
                timing_results = self.run_classification_benchmark()
            elif self.model_type == 'detection':
                timing_results = self.run_detection_benchmark()
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            return None
        except Exception as e:
            print(f"测试过程中出错: {e}")
            return None
        finally:
            self.monitoring = False
            time.sleep(0.5)  # 等待监控线程结束
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # 计算统计信息
        stats = self.calculate_statistics(timing_results, total_time)
        
        return stats

    def calculate_statistics(self, timing_results, total_time):
        """计算统计信息"""
        stats = {
            'system_info': {
                'hostname': socket.gethostname(),
                'device': self.device,
                'model_type': self.model_type,
                'model_name': self.model_info['name'],
                'dataset': self.dataset_name,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'performance': {
                'total_samples': self.total_samples,
                'total_time': total_time,
                'throughput': self.total_samples / total_time if total_time > 0 else 0,
                'avg_time_per_sample': (total_time / self.total_samples * 1000) if self.total_samples > 0 else 0
            },
            'timing': {},
            'resources': {
                'cpu': {
                    'min': np.min(self.cpu_usage) if self.cpu_usage else 0,
                    'max': np.max(self.cpu_usage) if self.cpu_usage else 0,
                    'avg': np.mean(self.cpu_usage) if self.cpu_usage else 0
                },
                'memory': {
                    'min': np.min(self.memory_usage) if self.memory_usage else 0,
                    'max': np.max(self.memory_usage) if self.memory_usage else 0,
                    'avg': np.mean(self.memory_usage) if self.memory_usage else 0
                }
            }
        }
        
        # 添加时间统计
        for key, times in timing_results.items():
            if times:
                stats['timing'][key] = {
                    'min': np.min(times),
                    'max': np.max(times),
                    'avg': np.mean(times),
                    'std': np.std(times)
                }
        
        # GPU统计
        if torch.cuda.is_available() and self.gpu_memory:
            stats['resources']['gpu'] = {
                'memory': {
                    'min': np.min(self.gpu_memory),
                    'max': np.max(self.gpu_memory),
                    'avg': np.mean(self.gpu_memory)
                },
                'utilization': {
                    'min': np.min(self.gpu_utilization) if self.gpu_utilization else 0,
                    'max': np.max(self.gpu_utilization) if self.gpu_utilization else 0,
                    'avg': np.mean(self.gpu_utilization) if self.gpu_utilization else 0
                }
            }
        
        return stats

    def print_concise_results(self, stats):
        """打印简洁的结果"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        # 基本信息
        print(f"Model: {stats['system_info']['model_name']}")
        print(f"Dataset: {stats['system_info']['dataset']}")
        print(f"Device: {stats['system_info']['device']}")
        
        # 性能指标
        print(f"\nPerformance:")
        print(f"  Samples processed: {stats['performance']['total_samples']}")
        print(f"  Total time: {stats['performance']['total_time']:.2f}s")
        print(f"  Throughput: {stats['performance']['throughput']:.2f} samples/sec")
        print(f"  Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms")
        
        # 时间分解
        if stats['timing']:
            print(f"\nTiming breakdown (ms):")
            for stage, data in stats['timing'].items():
                stage_name = stage.replace('_', ' ').title()
                print(f"  {stage_name}: {data['avg']:.2f} ± {data['std']:.2f}")
        
        # 资源使用
        print(f"\nResource utilization:")
        print(f"  CPU: {stats['resources']['cpu']['avg']:.1f}%")
        print(f"  Memory: {stats['resources']['memory']['avg']:.1f}%")
        
        if 'gpu' in stats['resources']:
            print(f"  GPU Memory: {stats['resources']['gpu']['memory']['avg']:.1f}%")
            print(f"  GPU Util: {stats['resources']['gpu']['utilization']['avg']:.1f}%")
        
        # 性能评级
        fps = stats['performance']['throughput']
        if self.model_type == 'classification':
            if fps > 100: rating = "Excellent 🟢"
            elif fps > 50: rating = "Good 🟡"
            elif fps > 10: rating = "Fair 🟠"
            else: rating = "Slow 🔴"
        else:  # detection
            if fps > 30: rating = "Excellent 🟢"
            elif fps > 15: rating = "Good 🟡"
            elif fps > 5: rating = "Fair 🟠"
            else: rating = "Slow 🔴"
        
        print(f"\nOverall Rating: {rating}")

    def save_csv_results(self, stats):
        """保存CSV结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        filename = f"{hostname}_{self.model_type}_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 基本信息
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Hostname', stats['system_info']['hostname'], ''])
            writer.writerow(['Model Type', stats['system_info']['model_type'], ''])
            writer.writerow(['Model Name', stats['system_info']['model_name'], ''])
            writer.writerow(['Dataset', stats['system_info']['dataset'], ''])
            writer.writerow(['Device', stats['system_info']['device'], ''])
            writer.writerow(['Device Name', stats['system_info']['device_name'], ''])
            
            # 性能指标
            writer.writerow(['Total Samples', stats['performance']['total_samples'], 'samples'])
            writer.writerow(['Total Time', f"{stats['performance']['total_time']:.2f}", 'seconds'])
            writer.writerow(['Throughput', f"{stats['performance']['throughput']:.2f}", 'samples/sec'])
            writer.writerow(['Avg Time per Sample', f"{stats['performance']['avg_time_per_sample']:.2f}", 'ms'])
            
            # 时间分解
            for stage, data in stats['timing'].items():
                writer.writerow([f'{stage.replace("_", " ").title()} Min', f"{data['min']:.2f}", 'ms'])
                writer.writerow([f'{stage.replace("_", " ").title()} Max', f"{data['max']:.2f}", 'ms'])
                writer.writerow([f'{stage.replace("_", " ").title()} Avg', f"{data['avg']:.2f}", 'ms'])
                writer.writerow([f'{stage.replace("_", " ").title()} Std', f"{data['std']:.2f}", 'ms'])
            
            # 资源使用
            writer.writerow(['CPU Usage Avg', f"{stats['resources']['cpu']['avg']:.1f}", '%'])
            writer.writerow(['Memory Usage Avg', f"{stats['resources']['memory']['avg']:.1f}", '%'])
            
            if 'gpu' in stats['resources']:
                writer.writerow(['GPU Memory Avg', f"{stats['resources']['gpu']['memory']['avg']:.1f}", '%'])
                writer.writerow(['GPU Utilization Avg', f"{stats['resources']['gpu']['utilization']['avg']:.1f}", '%'])
        
        print(f"\nCSV结果已保存至: {filename}")
        return filename

    def create_visualizations(self, stats, csv_filename):
        """创建可视化图表"""
        try:
            print("正在生成可视化图表...")
            
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Benchmark Results: {stats["system_info"]["model_name"]}', fontsize=16)
            
            # 1. 时间分解饼图
            if stats['timing']:
                timing_data = [(k.replace('_', ' ').title(), v['avg']) for k, v in stats['timing'].items()]
                labels, values = zip(*timing_data)
                ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Timing Breakdown')
            
            # 2. 资源利用率柱状图
            resources = ['CPU', 'Memory']
            usage = [stats['resources']['cpu']['avg'], stats['resources']['memory']['avg']]
            
            if 'gpu' in stats['resources']:
                resources.extend(['GPU Memory', 'GPU Util'])
                usage.extend([stats['resources']['gpu']['memory']['avg'], 
                             stats['resources']['gpu']['utilization']['avg']])
            
            bars = ax2.bar(resources, usage, color=['skyblue', 'lightgreen', 'orange', 'red'][:len(resources)])
            ax2.set_title('Resource Utilization')
            ax2.set_ylabel('Usage (%)')
            ax2.set_ylim(0, 100)
            
            # 添加数值标签
            for bar, val in zip(bars, usage):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.1f}%', ha='center', va='bottom')
            
            # 3. 性能对比（如果有多个时间阶段）
            if len(stats['timing']) > 1:
                stages = list(stats['timing'].keys())
                avg_times = [stats['timing'][stage]['avg'] for stage in stages]
                std_times = [stats['timing'][stage]['std'] for stage in stages]
                
                x_pos = np.arange(len(stages))
                ax3.bar(x_pos, avg_times, yerr=std_times, capsize=5, 
                       color='lightcoral', alpha=0.7)
                ax3.set_xlabel('Processing Stage')
                ax3.set_ylabel('Time (ms)')
                ax3.set_title('Processing Time by Stage')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels([s.replace('_', ' ').title() for s in stages], rotation=45)
            
            # 4. 系统信息文本
            system_text = f"""
System Information:
Device: {stats['system_info']['device']}
Model: {stats['system_info']['model_name']}
Dataset: {stats['system_info']['dataset']}

Performance Summary:
Throughput: {stats['performance']['throughput']:.2f} samples/sec
Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms
Total samples: {stats['performance']['total_samples']}
            """
            ax4.text(0.1, 0.5, system_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('System Info')
            
            plt.tight_layout()
            
            # 保存图表
            plot_filename = csv_filename.replace('.csv', '_visualization.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"可视化图表已保存至: {plot_filename}")
            
            # 显示图表（可选）
            # plt.show()
            
        except Exception as e:
            print(f"创建可视化时出错: {e}")

def main():
    """主函数"""
    # 检查依赖
    missing_deps = []
    
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics (pip install ultralytics)")
    
    if not NVML_AVAILABLE:
        missing_deps.append("nvidia-ml-py3 (pip install nvidia-ml-py3)")
    
    try:
        import timm
    except ImportError:
        missing_deps.append("timm (pip install timm)")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        missing_deps.append("matplotlib seaborn (pip install matplotlib seaborn)")
    
    if missing_deps:
        print("建议安装以下依赖以获得完整功能:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()
    
    # 创建基准测试工具
    benchmark = InteractiveBenchmark()
    
    # 交互式设置
    benchmark.interactive_setup()
    
    # 加载数据集
    benchmark.load_dataset_func()
    
    # 加载模型
    benchmark.load_model()
    
    # 运行基准测试
    stats = benchmark.run_benchmark()
    
    if stats:
        # 打印简洁结果
        benchmark.print_concise_results(stats)
        
        # 保存CSV结果
        csv_filename = benchmark.save_csv_results(stats)
        
        # 创建可视化
        benchmark.create_visualizations(stats, csv_filename)
        
        print("\n测试完成!")
    else:
        print("测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
