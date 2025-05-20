#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JENS火灾烟雾报警系统报警模块
负责处理检测到的火灾和烟雾情况，并触发相应的报警
"""

import os
import time
import datetime
import threading
import winsound  # Windows声音播放
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import cv2
import numpy as np

from JENS_utils.JENS_config import Config
from JENS_utils.JENS_logger import get_logger
from JENS_utils.JENS_database import JENSDatabase

# 导入资源管理模块
from JENS_gui.JENS_resources import Sounds

# 移除通知模块引用
NOTIFICATION_AVAILABLE = False

logger = get_logger("JENS_alarm")


class AlarmManager:
    """报警管理器类"""
    
    # 报警级别
    ALARM_LEVEL_NORMAL = 0   # 正常
    ALARM_LEVEL_WARNING = 1  # 警告
    ALARM_LEVEL_ALARM = 2    # 报警
    
    def __init__(self):
        """初始化报警管理器"""
        self.config = Config()
        self.db = JENSDatabase()
        
        # 获取报警配置
        alarm_config = self.config.get_config("alarm")
        
        # 报警参数
        self.alarm_threshold = alarm_config.get("alarm_threshold", 0.6)
        self.warning_threshold = alarm_config.get("warning_threshold", 0.4)
        self.alarm_frames_count = alarm_config.get("alarm_frames", 3)
        self.enable_sound = alarm_config.get("enable_sound", True)
        self.save_alarm_images = alarm_config.get("save_alarm_images", True)
        self.save_alarm_videos = alarm_config.get("save_alarm_videos", True)
        self.enable_notifications = alarm_config.get("enable_notifications", True)
        self.enable_auto_recovery = alarm_config.get("enable_auto_recovery", True)
        
        # 报警状态
        self.alarm_level = self.ALARM_LEVEL_NORMAL
        self.continuous_alarm_frames = 0
        self.alarm_active = False
        self.last_alarm_time = 0
        self.repeated_alarms_count = 0
        self.max_repeated_alarms = 5  # 最多连续报警次数
        self.auto_recovery_timeout = 300  # 自动恢复超时(秒)
        self.last_recovery_time = 0
        
        # 报警回调函数
        self.on_alarm_callback = None
        self.on_warning_callback = None
        self.on_normal_callback = None
        
        # 报警资源存储路径
        self._init_storage_dirs()
        
        # 报警声音播放标志和线程
        self.sound_playing = False
        self.sound_thread = None
        
        # 视频录制相关
        self.video_writer = None
        self.recording_frames = []
        self.recording_start_time = 0
        self.max_recording_length = alarm_config.get("video_length", 10)  # 秒
        
        # 自动恢复线程
        self.recovery_thread = None
        if self.enable_auto_recovery:
            self._start_auto_recovery_thread()
    
    def _init_storage_dirs(self):
        """初始化存储目录"""
        # 报警图片存储目录
        self.alarm_images_dir = Path("JENS_data/alarm_images")
        self.alarm_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 报警视频存储目录
        self.alarm_videos_dir = Path("JENS_data/alarm_videos")
        self.alarm_videos_dir.mkdir(parents=True, exist_ok=True)
        
        # 报警日志目录
        self.alarm_logs_dir = Path("JENS_data/alarm_logs")
        self.alarm_logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"已初始化报警资源存储目录")
    
    def _start_auto_recovery_thread(self):
        """启动自动恢复线程"""
        if self.recovery_thread is None or not self.recovery_thread.is_alive():
            self.recovery_thread = threading.Thread(target=self._auto_recovery_loop, daemon=True)
            self.recovery_thread.start()
            logger.info("自动恢复线程已启动")
    
    def _auto_recovery_loop(self):
        """自动恢复线程循环"""
        while self.enable_auto_recovery:
            current_time = time.time()
            
            # 如果当前处于报警状态，且超过恢复超时时间，则自动恢复
            if (self.alarm_level == self.ALARM_LEVEL_ALARM and 
                current_time - self.last_alarm_time > self.auto_recovery_timeout):
                logger.info(f"触发自动恢复机制，报警持续时间超过 {self.auto_recovery_timeout} 秒")
                self.reset_alarm()
                self.last_recovery_time = current_time
            
            # 休眠一段时间
            time.sleep(10)  # 每10秒检查一次
    
    def set_callbacks(self, on_alarm=None, on_warning=None, on_normal=None):
        """设置报警回调函数"""
        self.on_alarm_callback = on_alarm
        self.on_warning_callback = on_warning
        self.on_normal_callback = on_normal
    
    def check_detection(self, detections: List[Dict], frame: Optional[np.ndarray] = None, camera_id: Optional[int] = None) -> int:
        """
        检查检测结果并返回报警级别
        
        参数:
            detections (list): 检测结果列表
            frame (np.ndarray, 可选): 当前帧，用于保存报警图片
            camera_id (int, 可选): 摄像头ID
            
        返回:
            int: 报警级别
        """
        if not detections:
            self.continuous_alarm_frames = 0
            return self._update_alarm_level(self.ALARM_LEVEL_NORMAL)
        
        # 获取最高置信度和对应的类别
        max_confidence = 0
        alarm_class = ""
        for det in detections:
            if det['confidence'] > max_confidence:
                max_confidence = det['confidence']
                alarm_class = det['class_name']
        
        # 判断报警级别
        if max_confidence >= self.alarm_threshold:
            self.continuous_alarm_frames += 1
            
            # 连续多帧超过阈值时触发报警
            if self.continuous_alarm_frames >= self.alarm_frames_count:
                # 如果有当前帧且保存报警图片功能已启用，则保存图片
                saved_image_path = None
                if frame is not None and self.save_alarm_images:
                    saved_image_path = self._save_alarm_image(frame, alarm_class, max_confidence, camera_id)
                
                # 如果录制功能已启用，开始录制
                if frame is not None and self.save_alarm_videos:
                    self._start_recording(frame, camera_id)
                
                # 触发报警
                alarm_data = self.trigger_alarm(camera_id, detections, saved_image_path)
                
                return self._update_alarm_level(self.ALARM_LEVEL_ALARM)
            
            # 否则保持当前状态
            return self.alarm_level
            
        elif max_confidence >= self.warning_threshold:
            self.continuous_alarm_frames = 0
            
            # 触发警告
            self.trigger_warning(camera_id)
            
            return self._update_alarm_level(self.ALARM_LEVEL_WARNING)
        
        else:
            self.continuous_alarm_frames = 0
            return self._update_alarm_level(self.ALARM_LEVEL_NORMAL)
    
    def _update_alarm_level(self, level: int) -> int:
        """
        更新报警级别并执行回调
        
        参数:
            level (int): 新的报警级别
            
        返回:
            int: 更新后的报警级别
        """
        if level == self.alarm_level:
            return level
        
        old_level = self.alarm_level
        self.alarm_level = level
        
        # 从报警状态切换到非报警状态时，停止录制
        if old_level == self.ALARM_LEVEL_ALARM and level != self.ALARM_LEVEL_ALARM:
            self._stop_recording()
            # 重置连续报警计数
            self.repeated_alarms_count = 0
        
        # 记录状态变化
        logger.info(f"报警级别从 {old_level} 变为 {level}")
        
        # 执行回调
        if level == self.ALARM_LEVEL_ALARM and self.on_alarm_callback:
            self.on_alarm_callback()
        elif level == self.ALARM_LEVEL_WARNING and self.on_warning_callback:
            self.on_warning_callback()
        elif level == self.ALARM_LEVEL_NORMAL and self.on_normal_callback:
            self.on_normal_callback()
        
        return level
    
    def _save_alarm_image(self, frame, alarm_type, confidence, camera_id=None):
        """
        保存报警图片
        
        参数:
            frame: 报警帧
            alarm_type: 报警类型
            confidence: 置信度
            camera_id: 摄像头ID
        """
        try:
            # 生成文件名：时间戳_摄像头ID_类型_置信度.jpg
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_str = f"cam{camera_id}" if camera_id is not None else "cam0"
            filename = f"{timestamp}_{camera_str}_{alarm_type}_{confidence:.2f}.jpg"
            filepath = self.alarm_images_dir / filename
            
            # 保存图片
            cv2.imwrite(str(filepath), frame)
            logger.info(f"已保存报警图片: {filepath}")
            
            # 记录到数据库
            try:
                # 检查设备ID是否存在，如果不存在则创建一个默认设备
                device_id = camera_id or 1
                if device_id != 0:  # 0是保留ID，不使用
                    # 检查设备是否存在
                    device_exists = False
                    try:
                        device_info = self.db.execute_query(
                            "SELECT id FROM jens_devices WHERE id = %s", 
                            (device_id,)
                        )
                        device_exists = device_info and len(device_info) > 0
                    except Exception:
                        device_exists = False
                    
                    # 如果设备不存在，创建一个默认设备
                    if not device_exists:
                        try:
                            # 尝试创建默认设备
                            device_name = f"Camera {device_id}"
                            logger.info(f"创建默认设备: ID={device_id}, 名称={device_name}")
                            
                            self.db.insert('jens_devices', {
                                'id': device_id,
                                'device_name': device_name,
                                'device_type': 'camera',
                                'status': 1
                            })
                        except Exception as e:
                            logger.error(f"创建默认设备失败: {str(e)}")
                            # 使用ID=1作为备用
                            device_id = 1
                            # 检查ID=1是否存在
                            try:
                                device_info = self.db.execute_query(
                                    "SELECT id FROM jens_devices WHERE id = 1"
                                )
                                if not (device_info and len(device_info) > 0):
                                    # 创建ID=1的默认设备
                                    self.db.insert('jens_devices', {
                                        'id': 1,
                                        'device_name': 'Default Camera',
                                        'device_type': 'camera',
                                        'status': 1
                                    })
                            except Exception as ex:
                                logger.error(f"创建备用设备(ID=1)失败: {str(ex)}")
                                return str(filepath)  # 仅返回图片路径，不记录到数据库
                
                alarm_type_id = 1 if alarm_type.lower() == 'fire' else 2  # 1-火焰, 2-烟雾
                alarm_level = 3 if confidence > 0.85 else 2  # 根据置信度设置报警级别
                
                alarm_id = self.db.add_alarm_event(
                    device_id=device_id,
                    alarm_type=alarm_type_id,
                    confidence=confidence,
                    alarm_level=alarm_level,
                    image_path=str(filepath)
                )
                
                logger.info(f"报警事件已记录到数据库，ID={alarm_id}")
            except Exception as e:
                logger.error(f"记录报警事件到数据库失败: {str(e)}")
            
            return filepath
        except Exception as e:
            logger.error(f"保存报警图片失败: {str(e)}")
            return None
    
    def _start_recording(self, initial_frame, camera_id=None):
        """
        开始录制报警视频
        
        参数:
            initial_frame: 初始帧
            camera_id: 摄像头ID
        """
        try:
            # 检查是否已在录制
            if self.video_writer is not None:
                return
            
            # 生成文件名：时间戳_摄像头ID.mp4
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_str = f"cam{camera_id}" if camera_id is not None else "cam0"
            filename = f"{timestamp}_{camera_str}.mp4"
            filepath = self.alarm_videos_dir / filename
            
            # 获取视频参数
            height, width = initial_frame.shape[:2]
            
            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(filepath), fourcc, 20.0, (width, height))
            
            # 初始化录制缓冲区
            self.recording_frames = [initial_frame.copy()]
            self.recording_start_time = time.time()
            self.video_filepath = filepath
            self.recording_camera_id = camera_id
            
            logger.info(f"开始录制报警视频: {filepath}")
        except Exception as e:
            logger.error(f"启动视频录制失败: {str(e)}")
            self.video_writer = None
    
    def add_recording_frame(self, frame):
        """
        添加帧到录制缓冲区
        
        参数:
            frame: 视频帧
        """
        if self.video_writer is not None:
            try:
                # 添加帧到视频
                self.video_writer.write(frame)
                self.recording_frames.append(frame.copy())
                
                # 检查是否达到最大录制时长
                current_time = time.time()
                if current_time - self.recording_start_time >= self.max_recording_length:
                    self._stop_recording()
            except Exception as e:
                logger.error(f"添加视频帧失败: {str(e)}")
    
    def _stop_recording(self):
        """停止录制视频"""
        if self.video_writer is not None:
            try:
                # 释放视频写入器
                self.video_writer.release()
                
                # 更新数据库中的视频路径
                try:
                    # 获取最新报警记录
                    alarms = self.db.get_recent_alarms(limit=1)
                    if alarms and len(alarms) > 0:
                        alarm_id = alarms[0]['id']
                        self.db.update(
                            'jens_alarms',
                            {'video_path': str(self.video_filepath)},
                            {'id': alarm_id}
                        )
                except Exception as e:
                    logger.error(f"更新报警视频记录失败: {str(e)}")
                
                logger.info(f"已停止录制并保存视频: {self.video_filepath}")
                
                # 重置录制状态
                self.video_writer = None
                self.recording_frames = []
                self.recording_start_time = 0
            except Exception as e:
                logger.error(f"停止视频录制失败: {str(e)}")
    
    def trigger_alarm(self, camera_id: Optional[int] = None, detections: List[Dict] = None, image_path: Optional[str] = None) -> Dict:
        """
        触发报警
        
        参数:
            camera_id (int, 可选): 摄像头ID
            detections (list, 可选): 检测结果列表
            image_path (str, 可选): 保存的报警图像路径
            
        返回:
            Dict: 报警数据
        """
        # 避免频繁报警
        current_time = time.time()
        if current_time - self.last_alarm_time < 10:  # 10秒内不重复报警
            # 增加重复报警计数
            self.repeated_alarms_count += 1
            # 超过最大重复次数，自动清除报警状态
            if self.repeated_alarms_count > self.max_repeated_alarms:
                logger.warning(f"连续报警次数超过 {self.max_repeated_alarms} 次，可能为误报，自动清除报警状态")
                self.reset_alarm()
                self.repeated_alarms_count = 0
            return {}
        
        self.last_alarm_time = current_time
        self.alarm_active = True
        self.repeated_alarms_count = 1  # 重置为第一次报警
        
        # 构建报警数据
        alarm_data = {
            'alarm_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device_id': camera_id or 0,
            'alarm_type': 1 if detections and detections[0].get('class_name', '').lower() == 'fire' else 2,
            'confidence': max([d.get('confidence', 0) for d in detections]) if detections else 0,
            'image_path': image_path or '',
            'processed': False
        }
        
        # 记录报警
        logger.warning(f"触发报警! 摄像头ID: {camera_id}")
        
        # 播放报警声音
        if self.enable_sound and not self.sound_playing:
            self._play_alarm_sound()
        
        return alarm_data
    
    def trigger_warning(self, camera_id: Optional[int] = None):
        """
        触发警告
        
        参数:
            camera_id (int, 可选): 摄像头ID
        """
        # 记录警告
        logger.info(f"触发警告! 摄像头ID: {camera_id}")
        
        # 播放警告声音
        if self.enable_sound and not self.sound_playing:
            self._play_warning_sound()
    
    def reset_alarm(self):
        """重置报警状态"""
        if self.alarm_active:
            self.alarm_active = False
            self.continuous_alarm_frames = 0
            # 停止声音播放
            self._stop_sound()
            # 停止录制
            self._stop_recording()
            logger.info("报警状态已重置")
    
    def _play_alarm_sound(self):
        """播放报警声音"""
        self._play_sound(Sounds.ALARM)
    
    def _play_warning_sound(self):
        """播放警告声音"""
        self._play_sound(Sounds.WARNING)
    
    def _play_sound(self, sound_path):
        """
        播放指定的声音文件
        
        参数:
            sound_path: 声音文件路径
        """
        if not self.enable_sound or self.sound_playing:
            return
        
        # 检查声音文件是否存在
        if not os.path.exists(sound_path):
            logger.error(f"声音文件不存在: {sound_path}")
            return
        
        def sound_thread_func():
            try:
                self.sound_playing = True
                # 使用winsound播放声音
                winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                self.sound_playing = False
            except Exception as e:
                logger.error(f"播放声音失败: {str(e)}")
                self.sound_playing = False
        
        # 启动声音播放线程
        self.sound_thread = threading.Thread(target=sound_thread_func, daemon=True)
        self.sound_thread.start()
        logger.debug(f"正在播放声音: {os.path.basename(sound_path)}")
    
    def _stop_sound(self):
        """停止当前声音播放"""
        if self.sound_playing:
            # 停止声音
            winsound.PlaySound(None, winsound.SND_PURGE)
            self.sound_playing = False
            logger.debug("已停止声音播放")
    
    def get_alarm_level(self) -> int:
        """获取当前报警级别"""
        return self.alarm_level
    
    def set_alarm_threshold(self, threshold: float):
        """设置报警阈值"""
        self.alarm_threshold = threshold
        self.config.set("alarm_threshold", threshold)
        logger.info(f"报警阈值已设置为: {threshold}")
    
    def set_warning_threshold(self, threshold: float):
        """设置警告阈值"""
        self.warning_threshold = threshold
        self.config.set("warning_threshold", threshold)
        logger.info(f"警告阈值已设置为: {threshold}")
    
    def set_alarm_frames_count(self, count: int):
        """设置连续报警帧数"""
        self.alarm_frames_count = count
        self.config.set("alarm_frames", count)
        logger.info(f"连续报警帧数已设置为: {count}")
    
    def enable_sound_alarm(self, enable: bool = True):
        """启用/禁用声音报警"""
        self.enable_sound = enable
        self.config.set("enable_sound", enable)
        logger.info(f"声音报警已{'启用' if enable else '禁用'}")
    
    def enable_notifications(self, enable: bool = True):
        """启用/禁用通知功能"""
        self.enable_notifications = enable
        logger.info(f"通知功能已{'启用' if enable else '禁用'}")
    
    def enable_auto_recovery(self, enable: bool = True):
        """启用/禁用自动恢复功能"""
        self.enable_auto_recovery = enable
        self.config.set("enable_auto_recovery", enable)
        
        # 启动/停止自动恢复线程
        if enable and (self.recovery_thread is None or not self.recovery_thread.is_alive()):
            self._start_auto_recovery_thread()
        
        logger.info(f"自动恢复功能已{'启用' if enable else '禁用'}")
    
    def set_auto_recovery_timeout(self, timeout: int):
        """设置自动恢复超时时间（秒）"""
        self.auto_recovery_timeout = max(60, timeout)  # 最小60秒
        self.config.set("auto_recovery_timeout", self.auto_recovery_timeout)
        logger.info(f"自动恢复超时时间已设置为: {self.auto_recovery_timeout}秒")
    
    def set_max_recording_length(self, length: int):
        """设置最大录制时长（秒）"""
        self.max_recording_length = max(5, length)  # 最小5秒
        self.config.set("video_length", self.max_recording_length)
        logger.info(f"最大录制时长已设置为: {self.max_recording_length}秒")
    
    def get_recent_alarms(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的报警记录
        
        参数:
            limit (int): 返回记录的数量限制
            
        返回:
            list: 报警记录列表
        """
        try:
            return self.db.get_recent_alarms(limit=limit)
        except Exception as e:
            logger.error(f"获取报警记录失败: {str(e)}")
            return []
    
    def get_alarm_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        获取报警统计信息
        
        参数:
            days (int): 统计天数
            
        返回:
            Dict: 包含统计信息的字典
        """
        try:
            return self.db.get_alarm_stats(days=days)
        except Exception as e:
            logger.error(f"获取报警统计信息失败: {str(e)}")
            return {}
    
    def mark_alarm_processed(self, alarm_id: int, note: str = "", user: str = "system") -> bool:
        """
        标记报警为已处理
        
        参数:
            alarm_id (int): 报警ID
            note (str): 处理备注
            user (str): 处理用户
            
        返回:
            bool: 操作是否成功
        """
        try:
            self.db.update(
                'jens_alarms',
                {
                    'processed': 1,
                    'process_note': note,
                    'process_time': datetime.datetime.now(),
                    'process_user': user
                },
                {'id': alarm_id}
            )
            logger.info(f"已标记报警ID {alarm_id} 为已处理")
            return True
        except Exception as e:
            logger.error(f"标记报警为已处理失败: {str(e)}")
            return False
    
    def export_alarm_records(self, start_date: datetime.datetime, end_date: datetime.datetime, 
                           export_path: str) -> bool:
        """
        导出报警记录
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            export_path: 导出路径
            
        返回:
            bool: 是否成功导出
        """
        try:
            # 查询报警记录
            sql = """
            SELECT a.*, d.device_name 
            FROM jens_alarms a 
            LEFT JOIN jens_devices d ON a.device_id = d.id
            WHERE a.alarm_time BETWEEN %s AND %s
            ORDER BY a.alarm_time DESC
            """
            alarms = self.db.execute_query(sql, (start_date, end_date))
            
            if not alarms:
                logger.warning("没有找到符合条件的报警记录")
                return False
            
            # 导出为CSV格式
            import csv
            with open(export_path, 'w', newline='') as csvfile:
                fieldnames = ['id', 'alarm_time', 'device_name', 'alarm_type', 'confidence', 
                             'alarm_level', 'processed', 'process_time', 'process_user', 'process_note']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for alarm in alarms:
                    # 转换报警类型
                    if alarm.get('alarm_type') == 1:
                        alarm['alarm_type'] = '火灾'
                    elif alarm.get('alarm_type') == 2:
                        alarm['alarm_type'] = '烟雾'
                    
                    # 转换报警级别
                    if alarm.get('alarm_level') == 1:
                        alarm['alarm_level'] = '低'
                    elif alarm.get('alarm_level') == 2:
                        alarm['alarm_level'] = '中'
                    elif alarm.get('alarm_level') == 3:
                        alarm['alarm_level'] = '高'
                    
                    # 转换处理状态
                    alarm['processed'] = '已处理' if alarm.get('processed') == 1 else '未处理'
                    
                    writer.writerow(alarm)
            
            logger.info(f"已成功导出 {len(alarms)} 条报警记录到: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出报警记录失败: {str(e)}")
            return False


# 单例模式
_alarm_manager_instance = None

def get_alarm_manager() -> AlarmManager:
    """
    获取报警管理器单例
    
    返回:
        AlarmManager: 报警管理器实例
    """
    global _alarm_manager_instance
    
    if _alarm_manager_instance is None:
        _alarm_manager_instance = AlarmManager()
    
    return _alarm_manager_instance
