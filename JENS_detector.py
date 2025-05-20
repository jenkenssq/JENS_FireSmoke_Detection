#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JENS火灾烟雾检测器模块
使用YOLOv5实现火灾和烟雾的检测
"""

import os
import sys
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from threading import Thread, Lock
import pandas as pd

from JENS_utils.JENS_config import Config
from JENS_utils.JENS_logger import get_logger

logger = get_logger("JENS_detector")

class YOLODetector:
    """YOLOv5检测器类"""
    
    # 类别名称
    CLASSES = ['fire', 'smoke']
    
    # 类别颜色 (BGR格式)
    CLASS_COLORS = {
        'fire': (0, 0, 255),    # 红色
        'smoke': (128, 128, 128)  # 灰色
    }
    
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, device: str = ''):
        """
        初始化YOLOv5检测器
        
        参数:
            model_path (str): 模型文件路径
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
            device (str): 设备 ('cpu', 'cuda:0', 等)
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 自动选择设备
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self._load_model()
        
        # 运行参数
        self.frame_count = 0
        self.detection_freq = 5  # 每5帧检测一次
        self.last_detections = []
        self.last_result_frame = None
        
        # 高级检测参数
        self.enable_enhanced_detection = False  # 是否启用增强检测模式
        self.detection_history = []  # 检测历史记录，用于时序分析
        self.history_size = 5  # 保存历史记录的大小
        
        # 性能监控
        self.processing_times = []  # 记录处理时间
        self.max_times_history = 50  # 最多保存50次处理时间
        
        # 线程锁，用于多线程访问保护
        self.lock = Lock()
    
    def _load_model(self):
        """加载yolov5_best模型"""
        try:
            # 设置模型的绝对路径
            absolute_model_dir = "D:\\监控系统\\JENS_FireSmoke_Detection\\JENS_models"
            model_filename = "yolov5s_best.pt"
            self.model_path = os.path.join(absolute_model_dir, model_filename)
            
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            logger.info(f"正在加载模型: {self.model_path}, 设备: {self.device}")
            
            # 直接通过torch.hub加载YOLOv5模型
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.model_path,
                                      force_reload=False,
                                      trust_repo=True)
            self.model.conf = self.conf_thres
            self.model.iou = self.iou_thres
            self.model.to(self.device)
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"加载模型出错: {str(e)}")
            raise RuntimeError(f"无法加载模型: {str(e)}")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测图像中的火灾和烟雾
        
        参数:
            frame (np.ndarray): 输入的图像帧
            
        返回:
            tuple: (处理后的帧, 检测结果列表)
        """
        self.frame_count += 1
        
        # 每N帧进行一次检测
        if self.frame_count % self.detection_freq == 0:
            try:
                start_time = time.time()
                
                # 进行检测
                results = self.model(frame.copy())  # 使用复制的帧进行检测，避免原始帧被修改
                
                # 解析结果
                detections = self._process_results(results, frame.shape)
                
                # 时序分析（如果启用增强检测）
                if self.enable_enhanced_detection:
                    detections = self._analyze_detections_temporally(detections)
                
                # 在图像上绘制检测结果
                result_frame = frame.copy()  # 确保使用原始帧的副本
                result_frame = self._draw_detections(result_frame, detections)
                
                # 记录处理时间
                processing_time = time.time() - start_time
                with self.lock:
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > self.max_times_history:
                        self.processing_times.pop(0)
                
                # 保存结果用于后续帧
                with self.lock:
                    self.last_detections = detections
                    self.last_result_frame = result_frame.copy()  # 保存副本
                
                return result_frame, detections
            
            except Exception as e:
                logger.error(f"检测过程出错: {str(e)}")
                return frame.copy(), []  # 返回原始帧的副本
        
        # 如果不是检测帧，返回上一次的结果
        with self.lock:
            if self.last_result_frame is not None:
                if self.last_result_frame.shape == frame.shape:
                    return self.last_result_frame.copy(), self.last_detections.copy() if self.last_detections else []
        
        return frame.copy(), []  # 如果没有上一次的结果，返回原始帧的副本
    
    def _process_results(self, results, frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        处理YOLOv5的检测结果
        
        参数:
            results: YOLOv5检测结果
            frame_shape: 图像帧的形状 (高, 宽, 通道)
            
        返回:
            list: 检测结果列表
        """
        detections = []
        height, width = frame_shape[:2]
        
        try:
            # 处理基于ultralytics的结果
            if hasattr(results, 'pandas'):
                # ultralytics v8格式
                df = results.pandas().xyxy[0]
                
                if df.empty:
                    return []
                
                # 遍历检测结果
                for _, det in df.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    conf = float(det['confidence'])
                    cls = int(det['class'])
                    class_name = det['name']
                    
                    # 构建检测结果字典
                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2-x1, y2-y1),  # (x, y, w, h) 格式
                        'area': (x2-x1) * (y2-y1) / (width * height),  # 占比面积
                        'timestamp': time.time()  # 添加时间戳
                    }
                    
                    detections.append(detection)
            else:
                # 处理直接张量格式输出
                output = results.xyxy[0] if isinstance(results, tuple) else results[0]
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                    
                for *xyxy, conf, cls_id in output:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(cls_id)
                    class_name = self.CLASSES[cls] if cls < len(self.CLASSES) else f"class_{cls}"
                    
                    # 构建检测结果字典
                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2-x1, y2-y1),  # (x, y, w, h) 格式
                        'area': (x2-x1) * (y2-y1) / (width * height),  # 占比面积
                        'timestamp': time.time()  # 添加时间戳
                    }
                    
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"处理检测结果出错: {str(e)}")
        
        return detections
    
    def _analyze_detections_temporally(self, current_detections: List[Dict]) -> List[Dict]:
        """
        对检测结果进行时序分析，提高检测的准确性和稳定性
        
        参数:
            current_detections: 当前帧的检测结果
            
        返回:
            list: 过滤/增强后的检测结果列表
        """
        # 添加当前检测结果到历史记录
        with self.lock:
            self.detection_history.append(current_detections)
            # 保持历史记录长度
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            # 如果历史记录不足，直接返回当前检测结果
            if len(self.detection_history) < 3:
                return current_detections
        
        # 分析历史检测结果中的持续性目标
        # 这里实现一个简单的策略：如果目标在多帧中出现，提高其置信度
        enhanced_detections = []
        
        for det in current_detections:
            class_name = det['class_name']
            bbox = det['bbox']
            conf = det['confidence']
            
            # 检查此目标是否在之前的帧中也出现
            frames_appeared = 1  # 当前帧已经出现
            for past_detections in self.detection_history[:-1]:  # 不包括当前帧的记录
                for past_det in past_detections:
                    if past_det['class_name'] == class_name:
                        # 通过IoU检查是否是同一个目标
                        past_bbox = past_det['bbox']
                        iou = self._calculate_iou(bbox, past_bbox)
                        if iou > 0.5:  # 如果IoU大于阈值，认为是同一目标
                            frames_appeared += 1
                            break
            
            # 根据出现帧数调整置信度
            if frames_appeared > 1:
                # 根据连续出现的帧数增加置信度，最多增加0.15
                conf_boost = min(0.15, 0.05 * frames_appeared)
                new_conf = min(0.99, conf + conf_boost)
                det['confidence'] = new_conf
                det['boosted'] = True  # 标记为已增强
                logger.debug(f"增强检测: {class_name} 原置信度: {conf:.2f} 增强后: {new_conf:.2f}")
            
            enhanced_detections.append(det)
        
        return enhanced_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的IoU (Intersection over Union)
        
        参数:
            bbox1: 第一个边界框 (x, y, w, h)
            bbox2: 第二个边界框 (x, y, w, h)
            
        返回:
            float: IoU值
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 转换为(x1, y1, x2, y2)格式
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # 计算交集区域的坐标
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        # 计算交集区域面积
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算各自面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        参数:
            frame (np.ndarray): 原始图像帧
            detections (list): 检测结果列表
            
        返回:
            np.ndarray: 绘制了检测框的图像帧
        """
        # 确保使用frame的副本
        result_frame = frame.copy()
        
        for det in detections:
            # 从字典中获取检测框坐标
            x, y, w, h = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # 获取类别颜色
            color = self.CLASS_COLORS.get(det['class_name'], (0, 255, 0))
            
            # 如果是增强过的检测，显示不同的样式
            if det.get('boosted', False):
                thickness = 3  # 加粗边框
            else:
                thickness = 2  # 默认边框粗细
            
            # 绘制边界框
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, thickness)
            
            # 绘制标签背景
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_frame, (x, y-text_size[1]-5), (x+text_size[0], y), color, -1)
            
            # 绘制标签文本 (颜色反转以提高可读性)
            cv2.putText(result_frame, label, (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame
    
    def set_conf_thres(self, conf_thres: float):
        """设置置信度阈值"""
        self.conf_thres = conf_thres
        if hasattr(self.model, 'conf'):
            self.model.conf = conf_thres
    
    def set_iou_thres(self, iou_thres: float):
        """设置IoU阈值"""
        self.iou_thres = iou_thres
        if hasattr(self.model, 'iou'):
            self.model.iou = iou_thres
    
    def set_detection_freq(self, freq: int):
        """设置检测频率"""
        self.detection_freq = max(1, freq)  # 至少每帧检测一次
    
    def enable_enhanced_mode(self, enable: bool = True):
        """启用/禁用增强检测模式"""
        self.enable_enhanced_detection = enable
        logger.info(f"增强检测模式: {'已启用' if enable else '已禁用'}")
    
    def get_average_processing_time(self) -> float:
        """获取平均处理时间（秒）"""
        with self.lock:
            if self.processing_times:
                return sum(self.processing_times) / len(self.processing_times)
            return 0.0
    
    def change_model(self, new_model_path: str) -> bool:
        """
        切换到新的模型
        
        参数:
            new_model_path (str): 新模型的路径
            
        返回:
            bool: 是否成功切换模型
        """
        try:
            # 保存旧模型路径，以便切换失败时恢复
            old_model_path = self.model_path
            self.model_path = new_model_path
            
            # 尝试加载新模型
            self._load_model()
            
            logger.info(f"成功切换模型: {new_model_path}")
            return True
        except Exception as e:
            # 切换失败，恢复旧模型
            logger.error(f"切换模型失败: {str(e)}")
            
            # 恢复旧模型
            try:
                self.model_path = old_model_path
                self._load_model()
                logger.info("已恢复到原模型")
            except Exception as e2:
                logger.error(f"恢复原模型失败: {str(e2)}")
            
            return False


class AsyncDetector:
    """异步检测器类，在单独线程中运行检测器"""
    
    def __init__(self, detector: YOLODetector):
        """
        初始化异步检测器
        
        参数:
            detector (YOLODetector): YOLOv5检测器实例
        """
        self.detector = detector
        self.thread = None
        self.running = False
        self.lock = Lock()
        
        # 输入输出队列
        self.input_frame = None
        self.result_frame = None
        self.detections = []
        
        # 性能监控
        self.fps = 0.0
        self.last_update_time = time.time()
        self.frame_counter = 0
    
    def start(self):
        """启动异步检测线程"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            logger.info("异步检测线程已启动")
    
    def stop(self):
        """停止异步检测线程"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            logger.info("异步检测线程已停止")
    
    def _detection_loop(self):
        """检测线程主循环"""
        while self.running:
            # 检查是否有新帧需要处理
            with self.lock:
                frame = self.input_frame
                if frame is None:
                    time.sleep(0.01)  # 避免CPU空转
                    continue
                
                # 重置输入帧，表示已取走
                self.input_frame = None
            
            # 执行检测
            try:
                result_frame, detections = self.detector.detect(frame)
                
                # 更新结果
                with self.lock:
                    self.result_frame = result_frame
                    self.detections = detections
                    
                    # 更新FPS计数
                    self.frame_counter += 1
                    current_time = time.time()
                    elapsed = current_time - self.last_update_time
                    if elapsed > 1.0:  # 每秒更新一次
                        self.fps = self.frame_counter / elapsed
                        self.last_update_time = current_time
                        self.frame_counter = 0
            
            except Exception as e:
                logger.error(f"检测线程出错: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        处理一帧图像
        
        参数:
            frame (np.ndarray): 输入图像帧
            
        返回:
            tuple: (处理后的帧, 检测结果列表)
        """
        # 更新输入帧
        with self.lock:
            self.input_frame = frame.copy()
            
            # 返回当前结果
            if self.result_frame is not None and self.result_frame.shape == frame.shape:
                return self.result_frame.copy(), self.detections
        
        # 如果没有结果，返回原始帧和空检测结果
        return frame, []
    
    def get_fps(self) -> float:
        """获取当前处理帧率"""
        with self.lock:
            return self.fps


# 单例模式
_detector_instance = None
_async_detector_instance = None

def get_detector(model_path: Optional[str] = None) -> YOLODetector:
    """
    获取检测器单例
    
    参数:
        model_path (str, 可选): 模型文件路径
        
    返回:
        YOLODetector: 检测器实例
    """
    global _detector_instance
    
    if _detector_instance is None and model_path:
        config = Config()
        detector_config = config.get_config("detector")
        conf_thres = detector_config.get("confidence_threshold", 0.25)
        iou_thres = detector_config.get("iou_threshold", 0.45)
        
        try:
            _detector_instance = YOLODetector(
                model_path=model_path,
                conf_thres=conf_thres,
                iou_thres=iou_thres
            )
        except Exception as e:
            logger.error(f"创建检测器失败: {str(e)}")
            return None
    
    return _detector_instance

def get_async_detector(model_path: Optional[str] = None) -> AsyncDetector:
    """
    获取异步检测器单例
    
    参数:
        model_path (str, 可选): 模型文件路径
        
    返回:
        AsyncDetector: 异步检测器实例
    """
    global _detector_instance, _async_detector_instance
    
    if _async_detector_instance is None:
        detector = get_detector(model_path)
        if detector:
            _async_detector_instance = AsyncDetector(detector)
        else:
            return None
    
    return _async_detector_instance


if __name__ == "__main__":
    # 测试代码
    
    # 模型路径
    model_path = os.path.join("JENS_models", "yolov5s_best.pt")
    
    
      
    image_path = os.path.join("Test_images", "image2.jpg")
        
        # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        sys.exit(1)
        
        # 创建检测器实例
    detector = YOLODetector(model_path)
        
        # 启用增强模式
    detector.enable_enhanced_mode(True)
        
        # 进行检测
    result_img, detections = detector.detect(img)
        
        # 打印检测结果
    print(f"检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        print(f"目标 {i+1}: {det['class_name']}, 置信度: {det['confidence']:.2f}")
        
        # 显示结果
    cv2.imshow("Detection Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
