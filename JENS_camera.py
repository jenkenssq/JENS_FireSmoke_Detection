#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
摄像头和视频处理模块
负责管理各种视频输入源，包括本地摄像头、IP摄像头和视频文件
"""

import os
import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

from JENS_utils.JENS_config import Config
from JENS_utils.JENS_database import JENSDatabase
from JENS_utils.JENS_logger import get_logger

logger = get_logger("JENS_camera")

class VideoSource:
    """视频源基类"""
    def __init__(self, source_id, name="未命名"):
        self.source_id = source_id  # 设备ID或文件路径
        self.name = name            # 设备名称或文件名
        self.cap = None             # OpenCV视频捕获对象
        self.is_running = False     # 是否正在运行
        self.frame_width = 640      # 帧宽度
        self.frame_height = 480     # 帧高度
        self.fps = 30               # 帧率
        self.thread = None          # 捕获线程
        self.current_frame = None   # 当前帧
        self.frame_count = 0        # 帧计数
        self.start_time = 0         # 开始时间
        self.lock = threading.Lock()  # 线程锁
    
    def open(self):
        """打开视频源"""
        try:
            self.cap = cv2.VideoCapture(self.source_id)
            if not self.cap.isOpened():
                logger.error(f"无法打开视频源: {self.source_id}")
                return False
            
            # 获取视频属性
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"成功打开视频源: {self.name} ({self.frame_width}x{self.frame_height} @ {self.fps}fps)")
            return True
        except Exception as e:
            logger.error(f"打开视频源时出错: {str(e)}")
            return False
    
    def start(self):
        """开始视频捕获"""
        if self.is_running:
            return True
        
        if not self.open():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"视频源 {self.name} 已启动")
        return True
    
    def stop(self):
        """停止视频捕获"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info(f"视频源 {self.name} 已停止")
    
    def _capture_loop(self):
        """视频捕获循环"""
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"视频源 {self.name} 无法读取帧，尝试重新连接...")
                    self.reconnect()
                    continue
                
                with self.lock:
                    self.current_frame = frame
                    self.frame_count += 1
                
                # 适当休眠减少CPU使用
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"视频捕获循环出错: {str(e)}")
                self.reconnect()
    
    def reconnect(self):
        """重新连接视频源"""
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(1.0)  # 等待一秒后重连
            self.cap = cv2.VideoCapture(self.source_id)
            
            if not self.cap.isOpened():
                logger.error(f"重新连接视频源失败: {self.source_id}")
                return False
            
            logger.info(f"成功重新连接视频源: {self.name}")
            return True
        except Exception as e:
            logger.error(f"重新连接视频源时出错: {str(e)}")
            return False
    
    def get_frame(self):
        """获取当前帧"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def get_fps(self):
        """获取实时FPS"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0
    
    def set_resolution(self, width, height):
        """设置分辨率"""
        if not self.cap:
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 更新实际分辨率
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return True

class USBCamera(VideoSource):
    """USB摄像头类"""
    def __init__(self, camera_id=0, name="USB摄像头"):
        super().__init__(camera_id, name)
        self.camera_id = camera_id

class IPCamera(VideoSource):
    """IP摄像头类"""
    def __init__(self, ip_address, port=554, username="", password="", channel=1, name="IP摄像头"):
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password
        self.channel = channel
        
        # 构建RTSP URL
        rtsp_url = self._build_rtsp_url()
        super().__init__(rtsp_url, name)
    
    def _build_rtsp_url(self):
        """构建RTSP URL"""
        # 基本RTSP URL模板，可根据不同摄像头协议调整
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        
        return f"rtsp://{auth}{self.ip_address}:{self.port}/ch{self.channel}/main/av_stream"

class VideoFile(VideoSource):
    """视频文件类"""
    def __init__(self, file_path, name=None):
        self.file_path = file_path
        file_name = os.path.basename(file_path)
        super().__init__(file_path, name or file_name)
        self.total_frames = 0
        self.current_position = 0
        self.is_paused = False
    
    def open(self):
        """打开视频文件"""
        result = super().open()
        if result:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return result
    
    def _capture_loop(self):
        """重写捕获循环以支持暂停/继续和循环播放"""
        while self.is_running and self.cap and self.cap.isOpened():
            if not self.is_paused:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        # 视频结束，循环播放
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    with self.lock:
                        self.current_frame = frame
                        self.frame_count += 1
                        self.current_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    # 按实际帧率控制播放速度
                    if self.fps > 0:
                        time.sleep(1 / self.fps)
                    else:
                        time.sleep(0.033)  # 默认约30fps
                except Exception as e:
                    logger.error(f"视频文件捕获循环出错: {str(e)}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)  # 暂停时减少CPU占用
    
    def pause(self):
        """暂停视频"""
        self.is_paused = True
        logger.info(f"视频 {self.name} 已暂停")
    
    def resume(self):
        """继续播放视频"""
        self.is_paused = False
        logger.info(f"视频 {self.name} 继续播放")
    
    def seek(self, position_percent):
        """跳转到视频的指定位置(百分比)"""
        if not self.cap or self.total_frames <= 0:
            return False
        
        frame_pos = int(self.total_frames * position_percent / 100)
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_position = frame_pos
        
        logger.debug(f"视频 {self.name} 跳转到 {position_percent}% (帧 {frame_pos})")
        return True
    
    def get_position_percent(self):
        """获取当前播放位置(百分比)"""
        if self.total_frames <= 0:
            return 0
        return (self.current_position / self.total_frames) * 100

class CameraManager:
    """摄像头管理器"""
    def __init__(self):
        self.cameras = {}  # 摄像头字典 {camera_id: VideoSource}
        self.active_camera = 0  # 当前活动摄像头ID
        self.db = JENSDatabase()  # 数据库连接
        self.config = Config()  # 配置信息
        
        # 初始化存储目录
        self._init_storage_dirs()
        
        # 从数据库加载摄像头信息
        self.load_cameras_from_db()
    
    def _init_storage_dirs(self):
        """初始化存储目录"""
        storage_config = self.config.get_config("storage")
        
        # 确保报警图片目录存在
        self.images_dir = Path(storage_config.get("alarm_images_dir", "JENS_storage/images"))
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保报警视频目录存在
        self.videos_dir = Path(storage_config.get("alarm_videos_dir", "JENS_storage/videos"))
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"存储目录初始化完成: 图片={self.images_dir}, 视频={self.videos_dir}")
    
    def load_cameras_from_db(self):
        """从数据库加载摄像头信息"""
        try:
            devices = self.db.get_all_devices()
            for device in devices:
                if device['device_type'] == 'usb':
                    camera = USBCamera(
                        camera_id=int(device['ip_address'] or 0),  # USB摄像头ID存储在ip_address字段
                        name=device['device_name']
                    )
                elif device['device_type'] == 'ip':
                    camera = IPCamera(
                        ip_address=device['ip_address'],
                        port=device['port'] or 554,
                        username=device['username'],
                        password=device['password'],
                        name=device['device_name']
                    )
                else:
                    continue  # 跳过未知类型
                
                self.cameras[device['id']] = camera
                logger.info(f"从数据库加载摄像头: ID={device['id']}, 名称={device['device_name']}")
            
            logger.info(f"成功从数据库加载 {len(self.cameras)} 个摄像头")
        except Exception as e:
            logger.error(f"从数据库加载摄像头时出错: {str(e)}")
    
    def add_camera(self, name, camera_type, ip_address=None, port=None, username=None, password=None):
        """添加摄像头到数据库和管理器
        Args:
            name: 摄像头名称
            camera_type: 摄像头类型 'usb' 或 'ip'
            ip_address: IP地址(对于IP摄像头)或设备ID(对于USB摄像头)
            port: 端口号(对于IP摄像头)
            username: 用户名(对于IP摄像头)
            password: 密码(对于IP摄像头)
        Returns:
            新增摄像头的ID或None
        """
        try:
            # 添加到数据库
            device_id = self.db.add_device(
                device_name=name,
                device_type=camera_type,
                ip_address=str(ip_address) if ip_address is not None else None,
                port=port,
                username=username,
                password=password
            )
            
            # 创建摄像头对象
            if camera_type == 'usb':
                camera = USBCamera(
                    camera_id=int(ip_address or 0),
                    name=name
                )
            elif camera_type == 'ip':
                camera = IPCamera(
                    ip_address=ip_address,
                    port=port or 554,
                    username=username,
                    password=password,
                    name=name
                )
            else:
                logger.error(f"未知的摄像头类型: {camera_type}")
                return None
            
            # 添加到管理器
            self.cameras[device_id] = camera
            logger.info(f"成功添加摄像头: ID={device_id}, 名称={name}")
            
            return device_id
        except Exception as e:
            logger.error(f"添加摄像头时出错: {str(e)}")
            return None
    
    def remove_camera(self, camera_id):
        """从数据库和管理器中移除摄像头
        Args:
            camera_id: 摄像头ID
        Returns:
            是否成功
        """
        try:
            # 停止摄像头
            if camera_id in self.cameras:
                if self.active_camera == camera_id:
                    self.stop_active_camera()
                
                if self.cameras[camera_id].is_running:
                    self.cameras[camera_id].stop()
                
                # 从管理器中移除
                del self.cameras[camera_id]
            
            # 从数据库中删除
            self.db.delete("jens_devices", {"id": camera_id})
            
            logger.info(f"成功移除摄像头: ID={camera_id}")
            return True
        except Exception as e:
            logger.error(f"移除摄像头时出错: {str(e)}")
            return False
    
    def get_camera_list(self):
        """获取摄像头列表
        Returns:
            摄像头列表 [{id, name, type, status}, ...]
        """
        try:
            devices = self.db.get_all_devices()
            for device in devices:
                # 添加运行状态信息
                device['running'] = (device['id'] in self.cameras and 
                                     self.cameras[device['id']].is_running)
                device['active'] = (self.active_camera == device['id'])
            
            return devices
        except Exception as e:
            logger.error(f"获取摄像头列表时出错: {str(e)}")
            return []
    
    def set_active_camera(self, camera_id):
        """设置活动摄像头
        Args:
            camera_id: 摄像头ID
        Returns:
            是否成功
        """
        if camera_id not in self.cameras:
            logger.error(f"摄像头ID不存在: {camera_id}")
            return False
        
        if not self.cameras[camera_id].is_running:
            if not self.cameras[camera_id].start():
                logger.error(f"无法启动摄像头: ID={camera_id}")
                return False
        
        self.active_camera = camera_id
        logger.info(f"设置活动摄像头: ID={camera_id}, 名称={self.cameras[camera_id].name}")
        return True
    
    def stop_active_camera(self):
        """停止当前活动摄像头"""
        if self.active_camera and self.active_camera in self.cameras:
            self.cameras[self.active_camera].stop()
            logger.info(f"停止活动摄像头: ID={self.active_camera}")
            self.active_camera = None
            return True
        return False
    
    def get_frame(self):
        """获取当前活动摄像头的帧
        Returns:
            当前帧或None
        """
        if not self.active_camera or self.active_camera not in self.cameras:
            return None
        
        camera = self.cameras[self.active_camera]
        if not camera.is_running:
            return None
        
        return camera.get_frame()
    
    def get_active_camera_fps(self):
        """获取当前活动摄像头的FPS
        Returns:
            当前FPS或0.0
        """
        if not self.active_camera or self.active_camera not in self.cameras:
            return 0.0
        
        camera = self.cameras[self.active_camera]
        if not camera.is_running:
            return 0.0
        
        return camera.get_fps()
    
    def open_video_file(self, file_path):
        """打开视频文件
        Args:
            file_path: 视频文件路径
        Returns:
            视频对象或None
        """
        try:
            # 停止当前活动摄像头
            self.stop_active_camera()
            
            # 创建视频文件对象
            video = VideoFile(file_path)
            if not video.start():
                logger.error(f"无法打开视频文件: {file_path}")
                return None
            
            # 使用特殊ID存储视频文件
            self.cameras[-1] = video
            self.active_camera = -1
            
            logger.info(f"成功打开视频文件: {file_path}")
            return video
        except Exception as e:
            logger.error(f"打开视频文件时出错: {str(e)}")
            return None
    
    def save_alarm_image(self, frame, alarm_type, confidence, camera_id=None):
        """保存报警图像
        Args:
            frame: 图像帧
            alarm_type: 报警类型 1-火焰 2-烟雾
            confidence: 置信度
            camera_id: 摄像头ID，默认为当前活动摄像头
        Returns:
            保存的图像路径或None
        """
        if frame is None:
            return None
        
        try:
            camera_id = camera_id or self.active_camera
            if not camera_id:
                return None
            
            # 创建文件名
            alarm_type_str = "fire" if alarm_type == 1 else "smoke"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"alarm_{alarm_type_str}_{camera_id}_{timestamp}_{int(confidence*100)}.jpg"
            file_path = self.images_dir / file_name
            
            # 保存图像
            cv2.imwrite(str(file_path), frame)
            
            logger.info(f"报警图像已保存: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"保存报警图像时出错: {str(e)}")
            return None
    
    def start_alarm_recording(self, camera_id=None, duration=10):
        """开始报警录像
        Args:
            camera_id: 摄像头ID，默认为当前活动摄像头
            duration: 录像时长(秒)
        Returns:
            录像线程对象或None
        """
        # 此功能需要进一步实现
        pass

# 全局摄像头管理器实例
camera_manager = CameraManager()

def get_camera_manager():
    """获取摄像头管理器实例"""
    return camera_manager
