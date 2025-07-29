#!/usr/bin/env python3
"""
YOLOv8 性能基准测试工具 - 改进版
主要改进：
1. 支持命令行参数，不用修改代码
2. 支持多种输出格式（txt, json, csv）
3. 更好的错误处理
4. 自动检测文件和设备
5. 可选显示检测结果（像原版一样）
6. 可选显示进度信息
"""

import argparse
import json
import csv
from ultralytics import YOLO
import numpy as np
import time
import psutil
import torch
import threading
import socket
from collections import deque
import os
import sys
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available, GPU monitoring disabled")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 Performance Benchmark Tool')
    
    # 主要参数
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                        help='模型路径或名称 (默认: yolov8n.pt)')
    parser.add_argument('--source', '-s', default='0',
                        help='视频源: 文件路径或摄像头设备号 (默认: 0)')
    
    # 可选参数
    parser.add_argument('--device', '-d', default='auto',
                        help='运行设备: cpu, cuda:0, auto (默认: auto)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='批次大小 (默认: 1)')
    parser.add_argument('--max-frames', '-f', type=int,
                        help='最大处理帧数 (默认: 无限制)')
    parser.add_argument('--imgsz', type=int,
                        help='输入图像尺寸 (默认: 模型默认值)')
    
    # 输出选项
    parser.add_argument('--output-format', '-o', nargs='+', 
                        choices=['txt', 'json', 'csv'], default=['txt'],
                        help='输出格式 (默认: txt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    parser.add_argument('--show-detections', action='store_true',
                        help='显示每帧检测结果')
    parser.add_argument('--show-progress', action='store_true',
                        help='显示处理进度（每10帧）')
    
    return parser.parse_args()

class BenchmarkTool:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.monitoring = True
        
        # 系统信息
        self.cpu_count = psutil.cpu_count()
        self.cpu_logical_count = psutil.cpu_count(logical=True)
        
        # 初始化指标收集
        self.preprocess_times = []
        self.inference_times = []
        self.postprocess_times = []
        self.total_frames = 0
        self.start_time = None
        
        # 初始化资源监控
        self.cpu_percentages = deque(maxlen=1000)
        self.memory_usages = deque(maxlen=1000)
        self.gpu_mem_usages = deque(maxlen=1000) if torch.cuda.is_available() else None
        self.gpu_util_usages = deque(maxlen=1000) if torch.cuda.is_available() else None
        
        self.monitor_thread = None
        self.nvml_handle = None
    
    def print_detection_results(self, result, frame_num):
        """显示检测结果"""
        try:
            if result.boxes is not None and len(result.boxes) > 0:
                # 获取检测到的类别
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                # 统计每种物品的数量
                detections = {}
                for cls_id, conf in zip(class_ids, confidences):
                    class_name = result.names[cls_id]
                    if class_name not in detections:
                        detections[class_name] = []
                    detections[class_name].append(conf)
                
                # 格式化输出
                if detections:
                    detection_strs = []
                    for class_name, confs in detections.items():
                        count = len(confs)
                        avg_conf = sum(confs) / count
                        if count == 1:
                            detection_strs.append(f"{class_name}({avg_conf:.2f})")
                        else:
                            detection_strs.append(f"{class_name}x{count}({avg_conf:.2f})")
                    
                    print(f"Frame {frame_num}: {', '.join(detection_strs)}")
                else:
                    print(f"Frame {frame_num}: No objects detected")
            else:
                print(f"Frame {frame_num}: No objects detected")
                
        except Exception as e:
            if self.args.verbose:
                print(f"Frame {frame_num}: Detection display error: {e}")
            else:
                print(f"Frame {frame_num}: -")
        
    def load_model(self):
        """加载YOLO模型"""
        print(f"正在加载模型: {self.args.model}")
        try:
            self.model = YOLO(self.args.model)
            print(f"模型加载成功")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def validate_source(self):
        """验证输入源"""
        if self.args.source.isdigit():
            print(f"使用摄像头设备: {self.args.source}")
            return int(self.args.source)
        elif os.path.exists(self.args.source):
            print(f"使用视频文件: {self.args.source}")
            return self.args.source
        else:
            print(f"错误: 找不到输入源 '{self.args.source}'")
            sys.exit(1)
    
    def check_and_fix_device(self):
        """检查并修复设备设置"""
        if self.args.device == 'auto':
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # 测试CUDA是否真的可用
                try:
                    test_tensor = torch.tensor([1.0]).cuda()
                    self.args.device = 'cuda:0'
                    print(f"自动选择设备: cuda:0")
                except Exception as e:
                    print(f"CUDA不可用，自动切换到CPU: {e}")
                    self.args.device = 'cpu'
            else:
                print("未检测到可用的CUDA设备，使用CPU")
                self.args.device = 'cpu'
        
        # 验证设备
        if 'cuda' in self.args.device:
            try:
                test_tensor = torch.tensor([1.0]).to(self.args.device)
                print(f"设备验证成功: {self.args.device}")
            except Exception as e:
                print(f"设备 {self.args.device} 不可用，切换到CPU: {e}")
                self.args.device = 'cpu'

    def monitor_resources(self):
        """资源监控函数"""
        # 初始化GPU监控
        if torch.cuda.is_available() and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                if self.args.verbose:
                    print("GPU监控已初始化")
            except Exception as e:
                if self.args.verbose:
                    print(f"GPU监控初始化失败: {e}")
                self.nvml_handle = None
        
        # 获取当前进程
        current_process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU使用率
                self.cpu_percentages.append(current_process.cpu_percent(interval=0.1))
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self.memory_usages.append(memory.percent)
                
                # GPU使用率
                if torch.cuda.is_available():
                    # GPU内存使用率
                    gpu_mem_alloc = torch.cuda.memory_allocated(0)
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_mem_percent = (gpu_mem_alloc / gpu_mem_total) * 100
                    self.gpu_mem_usages.append(gpu_mem_percent)
                    
                    # GPU利用率
                    if self.nvml_handle is not None:
                        try:
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                            self.gpu_util_usages.append(utilization.gpu)
                        except Exception:
                            last_value = self.gpu_util_usages[-1] if self.gpu_util_usages else 0
                            self.gpu_util_usages.append(last_value)
                
                time.sleep(0.1)
                
            except Exception as e:
                if self.args.verbose:
                    print(f"监控错误: {e}")
                break
        
        # 清理
        if torch.cuda.is_available() and self.nvml_handle is not None and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def run_benchmark(self):
        """运行基准测试"""
        print("开始基准测试...")
        
        # 检查并修复设备设置
        self.check_and_fix_device()
        
        # 启动资源监控线程
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # 验证输入源
        source = self.validate_source()
        
        # 设置推理参数
        inference_params = {
            'stream': True,
            'batch': self.args.batch_size,
            'device': self.args.device,
            'verbose': False  # 避免干扰输出
        }
        
        if self.args.imgsz:
            inference_params['imgsz'] = self.args.imgsz
        
        if self.args.verbose:
            print(f"推理参数: {inference_params}")
            print(f"CPU配置: {self.cpu_count}核心/{self.cpu_logical_count}线程 (最大CPU使用率: {self.cpu_logical_count * 100}%)")
        
        self.start_time = time.time()
        
        # 用于计算瞬时FPS
        last_time = self.start_time
        last_frame_count = 0
        
        try:
            # 运行推理
            results = self.model(source, **inference_params)
            
            # 处理结果
            for result in results:
                # 检查帧数限制（如果不想限制帧数，可以注释掉下面几行）
                if self.args.max_frames and self.total_frames >= self.args.max_frames:
                    if self.args.verbose:
                        print(f"达到最大帧数限制: {self.args.max_frames}")
                    break
                
                # 收集指标
                speed_data = result.speed
                self.preprocess_times.append(speed_data.get("preprocess", 0))
                self.inference_times.append(speed_data.get("inference", 0))
                self.postprocess_times.append(speed_data.get("postprocess", 0))
                self.total_frames += 1
                
                # 显示检测结果（如果启用）
                if self.args.show_detections:
                    self.print_detection_results(result, self.total_frames)
                
                # 显示进度（显示累积平均FPS和瞬时FPS）
                if (self.args.show_progress or self.args.verbose) and self.total_frames % 10 == 0:
                    current_time = time.time()
                    
                    # 累积平均FPS
                    avg_fps = self.total_frames / (current_time - self.start_time)
                    
                    # 瞬时FPS（最近10帧的平均）
                    instant_fps = 10 / (current_time - last_time) if current_time > last_time else 0
                    
                    print(f"已处理 {self.total_frames} 帧 | 平均FPS: {avg_fps:.2f} | 瞬时FPS: {instant_fps:.2f}")
                    
                    last_time = current_time
                    last_frame_count = self.total_frames
                
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"推理过程出错: {e}")
            return False
        
        # 停止监控
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        if self.total_frames == 0:
            print("没有处理任何帧")
            return False
        
        print(f"测试完成，共处理 {self.total_frames} 帧")
        return True
    
    def calculate_statistics(self):
        """计算统计数据"""
        if not self.preprocess_times:
            return None
        
        end_time = time.time()
        total_time = end_time - self.start_time
        throughput = self.total_frames / total_time if total_time > 0 else 0
        
        total_per_frame = [p + i + pp for p, i, pp in zip(
            self.preprocess_times, self.inference_times, self.postprocess_times)]
        
        stats = {
            'summary': {
                'total_frames': self.total_frames,
                'total_time': total_time,
                'throughput': throughput,
                'avg_frame_time': np.mean(total_per_frame)
            },
            'timing': {
                'preprocess': {
                    'min': np.min(self.preprocess_times),
                    'max': np.max(self.preprocess_times),
                    'avg': np.mean(self.preprocess_times),
                    'std': np.std(self.preprocess_times)
                },
                'inference': {
                    'min': np.min(self.inference_times),
                    'max': np.max(self.inference_times),
                    'avg': np.mean(self.inference_times),
                    'std': np.std(self.inference_times)
                },
                'postprocess': {
                    'min': np.min(self.postprocess_times),
                    'max': np.max(self.postprocess_times),
                    'avg': np.mean(self.postprocess_times),
                    'std': np.std(self.postprocess_times)
                },
                'total_per_frame': {
                    'min': np.min(total_per_frame),
                    'max': np.max(total_per_frame),
                    'avg': np.mean(total_per_frame),
                    'std': np.std(total_per_frame)
                }
            },
            'resources': {
                'cpu': {
                    'min': np.min(self.cpu_percentages) if self.cpu_percentages else 0,
                    'max': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
                    'avg': np.mean(self.cpu_percentages) if self.cpu_percentages else 0
                },
                'memory': {
                    'min': np.min(self.memory_usages) if self.memory_usages else 0,
                    'max': np.max(self.memory_usages) if self.memory_usages else 0,
                    'avg': np.mean(self.memory_usages) if self.memory_usages else 0
                }
            },
            'system_info': {
                'hostname': socket.gethostname(),
                'model': self.args.model,
                'source': self.args.source,
                'device': self.args.device,
                'batch_size': self.args.batch_size,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'cpu_cores': self.cpu_count,
                'cpu_threads': self.cpu_logical_count,
                'cpu_max_usage': f"{self.cpu_logical_count * 100}%"
            }
        }
        
        # 添加GPU统计
        if torch.cuda.is_available() and self.gpu_mem_usages and self.gpu_util_usages:
            stats['resources']['gpu'] = {
                'memory': {
                    'min': np.min(self.gpu_mem_usages),
                    'max': np.max(self.gpu_mem_usages),
                    'avg': np.mean(self.gpu_mem_usages)
                },
                'utilization': {
                    'min': np.min(self.gpu_util_usages),
                    'max': np.max(self.gpu_util_usages),
                    'avg': np.mean(self.gpu_util_usages)
                }
            }
        
        return stats
    
    def save_results(self, stats):
        """保存结果到文件"""
        if not stats:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        base_filename = f"{hostname}_{timestamp}"
        
        # 保存TXT格式
        if 'txt' in self.args.output_format:
            txt_filename = f"{base_filename}.txt"
            self.save_txt_report(stats, txt_filename)
            print(f"TXT报告保存至: {txt_filename}")
        
        # 保存JSON格式
        if 'json' in self.args.output_format:
            json_filename = f"{base_filename}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"JSON报告保存至: {json_filename}")
        
        # 保存CSV格式
        if 'csv' in self.args.output_format:
            csv_filename = f"{base_filename}.csv"
            self.save_csv_report(stats, csv_filename)
            print(f"CSV报告保存至: {csv_filename}")
    
    def save_txt_report(self, stats, filename):
        """保存文本格式报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("===== YOLO性能基准测试报告 =====\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"主机名: {stats['system_info']['hostname']}\n")
            f.write(f"模型: {stats['system_info']['model']}\n")
            f.write(f"数据源: {stats['system_info']['source']}\n")
            f.write(f"设备: {stats['system_info']['device']}\n")
            f.write(f"批次大小: {stats['system_info']['batch_size']}\n")
            f.write(f"PyTorch版本: {stats['system_info']['torch_version']}\n")
            f.write(f"CUDA可用: {stats['system_info']['cuda_available']}\n")
            f.write(f"设备名称: {stats['system_info']['device_name']}\n\n")
            
            # 摘要
            f.write("===== 测试摘要 =====\n")
            f.write(f"总帧数: {stats['summary']['total_frames']}\n")
            f.write(f"总时间: {stats['summary']['total_time']:.2f} 秒\n")
            f.write(f"吞吐量: {stats['summary']['throughput']:.2f} FPS\n")
            f.write(f"平均帧时间: {stats['summary']['avg_frame_time']:.2f} 毫秒\n\n")
            
            # 时间详情
            f.write("===== 时间分解 =====\n")
            for stage, data in stats['timing'].items():
                stage_name = {'preprocess': '预处理', 'inference': '推理', 
                             'postprocess': '后处理', 'total_per_frame': '总计每帧'}
                f.write(f"{stage_name.get(stage, stage)}:\n")
                f.write(f"  最小值: {data['min']:.2f} 毫秒\n")
                f.write(f"  最大值: {data['max']:.2f} 毫秒\n")
                f.write(f"  平均值: {data['avg']:.2f} 毫秒\n")
                f.write(f"  标准差: {data['std']:.2f} 毫秒\n\n")
            
            # 资源使用
            f.write("===== 资源使用情况 =====\n")
            f.write(f"CPU使用率:\n")
            f.write(f"  最小值: {stats['resources']['cpu']['min']:.1f}%\n")
            f.write(f"  最大值: {stats['resources']['cpu']['max']:.1f}%\n")
            f.write(f"  平均值: {stats['resources']['cpu']['avg']:.1f}%\n\n")
            
            f.write(f"内存使用率:\n")
            f.write(f"  最小值: {stats['resources']['memory']['min']:.1f}%\n")
            f.write(f"  最大值: {stats['resources']['memory']['max']:.1f}%\n")
            f.write(f"  平均值: {stats['resources']['memory']['avg']:.1f}%\n\n")
            
            if 'gpu' in stats['resources']:
                f.write(f"GPU显存使用率:\n")
                f.write(f"  最小值: {stats['resources']['gpu']['memory']['min']:.1f}%\n")
                f.write(f"  最大值: {stats['resources']['gpu']['memory']['max']:.1f}%\n")
                f.write(f"  平均值: {stats['resources']['gpu']['memory']['avg']:.1f}%\n\n")
                
                f.write(f"GPU利用率:\n")
                f.write(f"  最小值: {stats['resources']['gpu']['utilization']['min']:.1f}%\n")
                f.write(f"  最大值: {stats['resources']['gpu']['utilization']['max']:.1f}%\n")
                f.write(f"  平均值: {stats['resources']['gpu']['utilization']['avg']:.1f}%\n\n")
    
    def save_csv_report(self, stats, filename):
        """保存CSV格式报告"""
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 表头
            writer.writerow(['指标', '数值', '单位'])
            
            # 摘要数据
            writer.writerow(['总帧数', stats['summary']['total_frames'], '帧'])
            writer.writerow(['总时间', f"{stats['summary']['total_time']:.2f}", '秒'])
            writer.writerow(['吞吐量', f"{stats['summary']['throughput']:.2f}", 'FPS'])
            writer.writerow(['平均帧时间', f"{stats['summary']['avg_frame_time']:.2f}", '毫秒'])
            
            # 推理时间
            writer.writerow(['推理最小时间', f"{stats['timing']['inference']['min']:.2f}", '毫秒'])
            writer.writerow(['推理最大时间', f"{stats['timing']['inference']['max']:.2f}", '毫秒'])
            writer.writerow(['推理平均时间', f"{stats['timing']['inference']['avg']:.2f}", '毫秒'])
            
            # 资源使用
            writer.writerow(['CPU平均使用率', f"{stats['resources']['cpu']['avg']:.1f}", '%'])
            writer.writerow(['内存平均使用率', f"{stats['resources']['memory']['avg']:.1f}", '%'])
            
            if 'gpu' in stats['resources']:
                writer.writerow(['GPU平均利用率', f"{stats['resources']['gpu']['utilization']['avg']:.1f}", '%'])
                writer.writerow(['GPU平均显存使用', f"{stats['resources']['gpu']['memory']['avg']:.1f}", '%'])
    
    def print_summary(self, stats):
        """打印详细摘要到控制台"""
        if not stats:
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total frames processed: {stats['summary']['total_frames']}")
        print(f"Total time elapsed: {stats['summary']['total_time']:.2f} seconds")
        print(f"Throughput: {stats['summary']['throughput']:.2f} frames per second")
        
        # 详细时间统计
        print("\n" + "="*60)
        print("DETAILED METRICS")
        print("="*60)
        print("Preprocess time (ms):")
        print(f"  Min: {stats['timing']['preprocess']['min']:.2f}")
        print(f"  Max: {stats['timing']['preprocess']['max']:.2f}")
        print(f"  Avg: {stats['timing']['preprocess']['avg']:.2f}")
        
        print("\nInference time (ms):")
        print(f"  Min: {stats['timing']['inference']['min']:.2f}")
        print(f"  Max: {stats['timing']['inference']['max']:.2f}")
        print(f"  Avg: {stats['timing']['inference']['avg']:.2f}")
        
        print("\nPostprocess time (ms):")
        print(f"  Min: {stats['timing']['postprocess']['min']:.2f}")
        print(f"  Max: {stats['timing']['postprocess']['max']:.2f}")
        print(f"  Avg: {stats['timing']['postprocess']['avg']:.2f}")
        
        print("\nTotal processing time per frame (ms):")
        print(f"  Min: {stats['timing']['total_per_frame']['min']:.2f}")
        print(f"  Max: {stats['timing']['total_per_frame']['max']:.2f}")
        print(f"  Avg: {stats['timing']['total_per_frame']['avg']:.2f}")
        
        # 资源使用统计
        print("\n" + "="*60)
        print("RESOURCE UTILIZATION")
        print("="*60)
        
        # CPU使用率解释
        cpu_avg = stats['resources']['cpu']['avg']
        cpu_cores = stats['system_info']['cpu_cores']
        cpu_threads = stats['system_info']['cpu_threads']
        cpu_utilization_percent = (cpu_avg / (cpu_threads * 100)) * 100
        
        print("CPU Usage:")
        print(f"  Min: {stats['resources']['cpu']['min']:.2f}%")
        print(f"  Max: {stats['resources']['cpu']['max']:.2f}%")
        print(f"  Avg: {stats['resources']['cpu']['avg']:.2f}%")
        print(f"  CPU Info: {cpu_cores}核心/{cpu_threads}线程 (最大{cpu_threads * 100}%)")
        print(f"  CPU利用率: {cpu_utilization_percent:.1f}% 的总CPU能力")
        
        if cpu_avg > 100:
            print(f"  说明: 使用了{cpu_avg/100:.1f}个CPU核心，这是正常的多核并行")
        
        print("\nMemory Usage (%):")
        print(f"  Min: {stats['resources']['memory']['min']:.2f}")
        print(f"  Max: {stats['resources']['memory']['max']:.2f}")
        print(f"  Avg: {stats['resources']['memory']['avg']:.2f}")
        
        if 'gpu' in stats['resources']:
            print("\nGPU Memory Usage (%):")
            print(f"  Min: {stats['resources']['gpu']['memory']['min']:.2f}")
            print(f"  Max: {stats['resources']['gpu']['memory']['max']:.2f}")
            print(f"  Avg: {stats['resources']['gpu']['memory']['avg']:.2f}")
            
            print("\nGPU Utilization (%):")
            print(f"  Min: {stats['resources']['gpu']['utilization']['min']:.2f}")
            print(f"  Max: {stats['resources']['gpu']['utilization']['max']:.2f}")
            print(f"  Avg: {stats['resources']['gpu']['utilization']['avg']:.2f}")
            print(f"  Device: {stats['system_info']['device_name']}")
        else:
            print("\nGPU: Not available")
        
        # 性能评级
        fps = stats['summary']['throughput']
        if fps > 30:
            rating = "Excellent 🟢"
        elif fps > 15:
            rating = "Good 🟡"
        elif fps > 5:
            rating = "Fair 🟠"
        else:
            rating = "Slow 🔴"
        
        print(f"\nPerformance Rating: {rating}")
        print(f"Model: {stats['system_info']['model']}")
        print(f"Device: {stats['system_info']['device']}")
        print(f"Batch Size: {stats['system_info']['batch_size']}")


def main():
    """主函数"""
    print("YOLOv8 性能基准测试工具 - 改进版")
    print("-" * 50)
    
    # 解析参数
    args = parse_arguments()
    
    # 创建基准测试工具
    benchmark = BenchmarkTool(args)
    
    # 加载模型
    benchmark.load_model()
    
    # 运行测试
    success = benchmark.run_benchmark()
    
    if success:
        # 计算统计数据
        stats = benchmark.calculate_statistics()
        
        # 打印摘要
        benchmark.print_summary(stats)
        
        # 保存结果
        benchmark.save_results(stats)
        
        print("\n测试完成!")
    else:
        print("测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
