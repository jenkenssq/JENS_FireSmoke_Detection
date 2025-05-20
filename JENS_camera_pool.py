#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JENS火灾烟雾报警系统多摄像头并行处理模块
优化多摄像头数据处理性能，提供高效的多摄像头管理
"""

import os
import time
import queue
import threading
import multiprocessing
from typing import Dict, List, Optional, Tuple, Callable, Any
import cv2
import numpy as np

from JENS_utils.JENS_config import Config
from JENS_utils.JENS_logger import get_logger
from JENS_camera import CameraManager, get_camera_manager, VideoSource
from JENS_detector import YOLODetector, get_detector, AsyncDetector

logger = get_logger("JENS_camera_pool")

class CameraWorker:
    """摄像头工作线程类，负责单个摄像头的处理"""
    
    def __init__(self, camera_id: int, camera_source: VideoSource, detector: YOLODetector):
        """
        初始化摄像头工作线程
        
        参数:
            camera_id: 摄像头ID
            camera_source: 摄像头视频源
            detector: 检测器实例
        """
        self.camera_id = camera_id
        self.camera_source = camera_source
        self.detector = detector
        
        # 状态信息
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        self.detection_count = 0
        self.last_detections = []
        self.last_frame = None
        self.result_frame = None
        self.last_update_time = time.time()
        
        # 线程
        self.worker_thread = None
        
        # 线程锁
        self.lock = threading.Lock()
    
    def start(self):
        """启动工作线程"""
        if self.is_running:
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"摄像头 {self.camera_id} 工作线程已启动")
    
    def stop(self):
        """停止工作线程"""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        logger.info(f"摄像头 {self.camera_id} 工作线程已停止")
    
    def _process_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 获取一帧图像
                success, frame = self.camera_source.read()
                
                if not success or frame is None:
                    time.sleep(0.01)  # 短暂休眠，避免CPU空转
                    continue
                
                # 更新计数和时间
                with self.lock:
                    self.frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - self.last_update_time
                    
                    if elapsed > 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.last_update_time = current_time
                
                # 处理图像（检测）
                result_frame, detections = self.detector.detect(frame)
                
                # 更新结果
                with self.lock:
                    if detections:
                        self.detection_count += 1
                    self.last_detections = detections
                    self.last_frame = frame
                    self.result_frame = result_frame
                
            except Exception as e:
                logger.error(f"摄像头 {self.camera_id} 处理出错: {str(e)}")
                time.sleep(0.1)  # 出错时休眠，避免连续错误
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        获取处理后的帧和检测结果
        
        返回:
            tuple: (处理后的帧, 检测结果列表)
        """
        with self.lock:
            if self.result_frame is not None:
                return self.result_frame.copy(), self.last_detections
            elif self.last_frame is not None:
                return self.last_frame.copy(), []
            else:
                return None, []
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        with self.lock:
            return self.fps
    
    def get_detection_count(self) -> int:
        """获取检测次数"""
        with self.lock:
            return self.detection_count


class CameraPool:
    """
    摄像头池，管理多个摄像头并行处理
    """
    
    def __init__(self, max_workers: int = None):
        """
        初始化摄像头池
        
        参数:
            max_workers: 最大工作线程数，默认为CPU核心数
        """
        self.config = Config()
        self.camera_manager = get_camera_manager()
        
        # 设置最大工作线程数
        if not max_workers:
            # 默认使用逻辑CPU核心数（如果有4核以上，留1核给其他任务）
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max(1, cpu_count - 1) if cpu_count > 4 else cpu_count
        else:
            self.max_workers = max_workers
        
        # 工作线程字典 {camera_id: CameraWorker}
        self.workers = {}
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 活动摄像头列表
        self.active_cameras = []
        
        # 检测器实例
        self.detector = None
        
        logger.info(f"摄像头池初始化完成，最大工作线程数: {self.max_workers}")
    
    def initialize(self):
        """初始化摄像头池，加载检测器"""
        try:
            # 获取检测器实例
            self.detector = get_detector()
            if not self.detector:
                from JENS_detector import YOLODetector
                model_path = os.path.join("JENS_models", "yolov5s.pt")
                self.detector = YOLODetector(model_path)
                
            logger.info("检测器已加载")
            return True
        except Exception as e:
            logger.error(f"初始化摄像头池失败: {str(e)}")
            return False
    
    def add_camera(self, camera_id: int) -> bool:
        """
        添加摄像头到池中并启动处理
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            bool: 是否成功添加
        """
        try:
            with self.lock:
                # 检查是否已存在
                if camera_id in self.workers:
                    logger.info(f"摄像头 {camera_id} 已在处理池中")
                    return True
                
                # 检查工作线程数是否超出限制
                if len(self.workers) >= self.max_workers:
                    logger.warning(f"工作线程数已达到最大值 {self.max_workers}，无法添加更多摄像头")
                    return False
                
                # 获取摄像头实例
                camera = self.camera_manager.get_camera(camera_id)
                if not camera:
                    logger.error(f"无法获取摄像头 {camera_id}")
                    return False
                
                # 确保检测器已初始化
                if not self.detector:
                    if not self.initialize():
                        return False
                
                # 创建工作线程
                worker = CameraWorker(camera_id, camera, self.detector)
                self.workers[camera_id] = worker
                
                # 启动工作线程
                worker.start()
                
                # 添加到活动摄像头列表
                if camera_id not in self.active_cameras:
                    self.active_cameras.append(camera_id)
                
                logger.info(f"摄像头 {camera_id} 已添加到处理池并启动")
                return True
                
        except Exception as e:
            logger.error(f"添加摄像头 {camera_id} 失败: {str(e)}")
            return False
    
    def remove_camera(self, camera_id: int) -> bool:
        """
        从池中移除摄像头并停止处理
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            bool: 是否成功移除
        """
        try:
            with self.lock:
                if camera_id not in self.workers:
                    logger.warning(f"摄像头 {camera_id} 不在处理池中")
                    return False
                
                # 停止工作线程
                worker = self.workers[camera_id]
                worker.stop()
                
                # 从字典中移除
                del self.workers[camera_id]
                
                # 从活动列表中移除
                if camera_id in self.active_cameras:
                    self.active_cameras.remove(camera_id)
                
                logger.info(f"摄像头 {camera_id} 已从处理池中移除")
                return True
                
        except Exception as e:
            logger.error(f"移除摄像头 {camera_id} 失败: {str(e)}")
            return False
    
    def get_frame(self, camera_id: int) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        获取指定摄像头的处理结果
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            tuple: (处理后的帧, 检测结果列表)
        """
        try:
            with self.lock:
                if camera_id not in self.workers:
                    return None, []
                
                worker = self.workers[camera_id]
                return worker.get_frame()
                
        except Exception as e:
            logger.error(f"获取摄像头 {camera_id} 帧失败: {str(e)}")
            return None, []
    
    def get_all_frames(self) -> Dict[int, Tuple[Optional[np.ndarray], List[Dict]]]:
        """
        获取所有摄像头的处理结果
        
        返回:
            dict: {camera_id: (处理后的帧, 检测结果列表)}
        """
        result = {}
        try:
            with self.lock:
                for camera_id, worker in self.workers.items():
                    result[camera_id] = worker.get_frame()
        except Exception as e:
            logger.error(f"获取所有摄像头帧失败: {str(e)}")
        
        return result
    
    def get_active_cameras(self) -> List[int]:
        """
        获取当前活动的摄像头ID列表
        
        返回:
            list: 摄像头ID列表
        """
        with self.lock:
            return self.active_cameras.copy()
    
    def get_worker_stats(self) -> Dict[int, Dict]:
        """
        获取所有工作线程的统计信息
        
        返回:
            dict: {camera_id: 统计信息字典}
        """
        stats = {}
        try:
            with self.lock:
                for camera_id, worker in self.workers.items():
                    stats[camera_id] = {
                        'fps': worker.get_fps(),
                        'detection_count': worker.get_detection_count()
                    }
        except Exception as e:
            logger.error(f"获取工作线程统计信息失败: {str(e)}")
        
        return stats
    
    def stop_all(self):
        """停止所有工作线程"""
        try:
            with self.lock:
                camera_ids = list(self.workers.keys())
                
                for camera_id in camera_ids:
                    self.remove_camera(camera_id)
                
                self.active_cameras = []
                
            logger.info("已停止所有摄像头处理线程")
            
        except Exception as e:
            logger.error(f"停止所有工作线程失败: {str(e)}")


# 单例模式
_camera_pool_instance = None

def get_camera_pool() -> CameraPool:
    """
    获取摄像头池单例
    
    返回:
        CameraPool: 摄像头池实例
    """
    global _camera_pool_instance
    
    if _camera_pool_instance is None:
        _camera_pool_instance = CameraPool()
        # 初始化摄像头池
        _camera_pool_instance.initialize()
        
    return _camera_pool_instance 