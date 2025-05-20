#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JENS火灾烟雾报警系统监控模块
负责监控系统状态、资源使用情况，并提供自动恢复功能
"""

import os
import time
import psutil
import platform
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta

from JENS_utils.JENS_config import Config
from JENS_utils.JENS_logger import get_logger

logger = get_logger("JENS_monitor")

class SystemMonitor:
    """系统监控类"""
    
    # 系统健康状态级别
    STATUS_NORMAL = 0      # 正常
    STATUS_WARNING = 1     # 警告
    STATUS_CRITICAL = 2    # 严重
    
    def __init__(self):
        """初始化系统监控器"""
        self.config = Config()
        
        # 监控配置
        self.monitor_config = self.config.get_config('monitor')
        self.enable_monitoring = self.monitor_config.get('enable', True)
        self.monitoring_interval = self.monitor_config.get('interval', 60)  # 秒
        self.log_interval = self.monitor_config.get('log_interval', 300)  # 秒
        
        # 阈值配置
        self.thresholds = self.monitor_config.get('thresholds', {
            'cpu_warning': 80,       # CPU使用率警告阈值 (%)
            'cpu_critical': 95,      # CPU使用率严重阈值 (%)
            'memory_warning': 80,    # 内存使用率警告阈值 (%)
            'memory_critical': 90,   # 内存使用率严重阈值 (%)
            'disk_warning': 85,      # 磁盘使用率警告阈值 (%)
            'disk_critical': 95,     # 磁盘使用率严重阈值 (%)
        })
        
        # 自动恢复配置
        self.recovery_config = self.monitor_config.get('recovery', {
            'enable': True,          # 启用自动恢复
            'max_cpu': 95,           # 触发恢复的CPU使用率上限
            'max_memory': 90,        # 触发恢复的内存使用率上限
            'min_free_space': 1024,  # 最小可用磁盘空间(MB)
            'recovery_actions': ['restart_detector', 'clear_memory', 'restart_system']
        })
        
        # 资源使用历史
        self.resource_history = {
            'cpu': [],           # CPU使用率历史
            'memory': [],        # 内存使用率历史
            'disk': [],          # 磁盘使用率历史
            'timestamp': []      # 时间戳
        }
        self.max_history_size = 100  # 最多保存100个历史记录点
        
        # 监控状态
        self.system_status = self.STATUS_NORMAL
        self.last_log_time = 0
        self.critical_event_count = 0
        self.max_critical_events = 3  # 连续严重事件阈值
        
        # 监控线程
        self.monitor_thread = None
        self.is_monitoring = False
        
        logger.info("系统监控器初始化完成")
    
    def start_monitoring(self):
        """启动系统监控"""
        if not self.enable_monitoring:
            logger.info("系统监控已禁用，不会启动")
            return
        
        if self.is_monitoring:
            logger.info("监控已在运行中")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止系统监控"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控线程主循环"""
        while self.is_monitoring:
            try:
                # 检查系统状态
                status = self.check_system_status()
                
                # 记录检查时间
                current_time = time.time()
                
                # 添加到历史记录
                self._add_to_history(status)
                
                # 定期记录日志
                if current_time - self.last_log_time >= self.log_interval:
                    self._log_system_status(status)
                    self.last_log_time = current_time
                
                # 如果状态为严重，执行恢复措施
                if status['status'] == self.STATUS_CRITICAL:
                    self.critical_event_count += 1
                    if self.critical_event_count >= self.max_critical_events:
                        logger.warning(f"检测到连续 {self.critical_event_count} 次严重状态，开始执行恢复措施")
                        self._perform_recovery(status)
                        self.critical_event_count = 0  # 重置计数
                else:
                    self.critical_event_count = 0  # 如果不是严重状态，重置计数
            
            except Exception as e:
                logger.error(f"监控过程中出错: {str(e)}")
            
            # 休眠指定时间
            time.sleep(self.monitoring_interval)
    
    def check_system_status(self) -> Dict:
        """
        检查系统状态
        
        返回:
            Dict: 包含系统状态信息的字典
        """
        try:
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 获取内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 获取磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 确定系统状态
            status = self.STATUS_NORMAL
            status_reason = []
            
            # 检查CPU
            if cpu_percent >= self.thresholds['cpu_critical']:
                status = max(status, self.STATUS_CRITICAL)
                status_reason.append(f"CPU使用率过高: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                status = max(status, self.STATUS_WARNING)
                status_reason.append(f"CPU使用率较高: {cpu_percent:.1f}%")
                
            # 检查内存
            if memory_percent >= self.thresholds['memory_critical']:
                status = max(status, self.STATUS_CRITICAL)
                status_reason.append(f"内存使用率过高: {memory_percent:.1f}%")
            elif memory_percent >= self.thresholds['memory_warning']:
                status = max(status, self.STATUS_WARNING)
                status_reason.append(f"内存使用率较高: {memory_percent:.1f}%")
                
            # 检查磁盘
            if disk_percent >= self.thresholds['disk_critical']:
                status = max(status, self.STATUS_CRITICAL)
                status_reason.append(f"磁盘使用率过高: {disk_percent:.1f}%")
            elif disk_percent >= self.thresholds['disk_warning']:
                status = max(status, self.STATUS_WARNING)
                status_reason.append(f"磁盘使用率较高: {disk_percent:.1f}%")
                
            # 检查磁盘剩余空间
            disk_free_mb = disk.free / (1024 * 1024)  # 转换为MB
            if disk_free_mb < self.recovery_config.get('min_free_space', 1024):
                status = max(status, self.STATUS_CRITICAL)
                status_reason.append(f"磁盘剩余空间不足: {disk_free_mb:.1f}MB")
            
            # 记录状态变化
            if status != self.system_status:
                if status > self.system_status:
                    logger.warning(f"系统状态从 {self.system_status} 提升到 {status}, 原因: {', '.join(status_reason)}")
                else:
                    logger.info(f"系统状态从 {self.system_status} 降低到 {status}")
                self.system_status = status
            
            # 返回状态信息
            return {
                'timestamp': datetime.now(),
                'status': status,
                'status_reason': status_reason,
                'cpu': cpu_percent,
                'memory': memory_percent,
                'disk': disk_percent,
                'disk_free_mb': disk_free_mb
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {str(e)}")
            return {
                'timestamp': datetime.now(),
                'status': self.STATUS_WARNING,
                'status_reason': [f"获取系统状态失败: {str(e)}"],
                'cpu': 0,
                'memory': 0,
                'disk': 0,
                'disk_free_mb': 0
            }
    
    def _add_to_history(self, status: Dict):
        """
        添加状态到历史记录
        
        参数:
            status: 系统状态信息
        """
        self.resource_history['timestamp'].append(status['timestamp'])
        self.resource_history['cpu'].append(status['cpu'])
        self.resource_history['memory'].append(status['memory'])
        self.resource_history['disk'].append(status['disk'])
        
        # 限制历史记录大小
        if len(self.resource_history['timestamp']) > self.max_history_size:
            self.resource_history['timestamp'].pop(0)
            self.resource_history['cpu'].pop(0)
            self.resource_history['memory'].pop(0)
            self.resource_history['disk'].pop(0)
    
    def _log_system_status(self, status: Dict):
        """
        记录系统状态到日志
        
        参数:
            status: 系统状态信息
        """
        status_text = "正常" if status['status'] == self.STATUS_NORMAL else "警告" if status['status'] == self.STATUS_WARNING else "严重"
        reason_text = ", ".join(status['status_reason']) if status['status_reason'] else "无"
        
        logger.info(f"系统状态: {status_text}, CPU: {status['cpu']:.1f}%, 内存: {status['memory']:.1f}%, "
                   f"磁盘: {status['disk']:.1f}%, 剩余空间: {status['disk_free_mb']:.1f}MB, 原因: {reason_text}")
    
    def _perform_recovery(self, status: Dict):
        """
        执行恢复措施
        
        参数:
            status: 系统状态信息
        """
        if not self.recovery_config.get('enable', True):
            logger.info("自动恢复功能已禁用，不执行恢复措施")
            return
        
        actions = self.recovery_config.get('recovery_actions', [])
        
        for action in actions:
            try:
                if action == 'restart_detector':
                    self._restart_detector()
                elif action == 'clear_memory':
                    self._clear_memory()
                elif action == 'cleanup_disk':
                    self._cleanup_disk()
                elif action == 'restart_system':
                    # 最后尝试重启系统，这是最严重的恢复措施
                    if status['cpu'] > self.recovery_config.get('max_cpu', 95) or \
                       status['memory'] > self.recovery_config.get('max_memory', 90) or \
                       status['disk_free_mb'] < self.recovery_config.get('min_free_space', 1024):
                        self._restart_system()
            except Exception as e:
                logger.error(f"执行恢复措施 {action} 失败: {str(e)}")
    
    def _restart_detector(self):
        """重启检测器模块"""
        logger.info("正在重启检测器模块...")
        try:
            from JENS_detector import get_detector
            detector = get_detector()
            if detector:
                # 重新加载模型
                detector.change_model(detector.model_path)
                logger.info("检测器模块重启成功")
                return True
            else:
                logger.warning("无法获取检测器实例")
                return False
        except Exception as e:
            logger.error(f"重启检测器模块失败: {str(e)}")
            return False
    
    def _clear_memory(self):
        """清理内存"""
        logger.info("正在清理内存...")
        
        try:
            # 在Windows上使用空操作
            if platform.system() == 'Windows':
                import ctypes
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
                
            # 在Linux上清理内存缓存
            elif platform.system() == 'Linux':
                # 建议系统清理缓存
                os.system("sync")
                
                # 尝试写入到/proc/sys/vm/drop_caches
                # 注意：需要root权限
                try:
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('3')
                except:
                    pass
                    
            # 强制执行Python垃圾回收
            import gc
            gc.collect()
            
            logger.info("内存清理完成")
            return True
            
        except Exception as e:
            logger.error(f"清理内存失败: {str(e)}")
            return False
    
    def _cleanup_disk(self):
        """清理磁盘空间"""
        logger.info("正在清理磁盘空间...")
        
        try:
            # 清理临时文件
            temp_dirs = []
            
            if platform.system() == 'Windows':
                # Windows临时目录
                temp_dirs = [os.environ.get('TEMP'), os.environ.get('TMP')]
            else:
                # Linux/Mac临时目录
                temp_dirs = ['/tmp']
            
            # 添加系统自己的临时目录
            system_temp = Path("JENS_temp")
            if system_temp.exists():
                temp_dirs.append(str(system_temp))
            
            # 清理日志目录中较旧的日志
            logs_dir = Path("logs")
            if logs_dir.exists():
                # 删除7天前的日志
                cutoff_date = datetime.now() - timedelta(days=7)
                for log_file in logs_dir.glob("*.log.*"):
                    try:
                        if log_file.stat().st_mtime < cutoff_date.timestamp():
                            log_file.unlink()
                            logger.debug(f"已删除旧日志文件: {log_file}")
                    except Exception as e:
                        logger.error(f"删除日志文件 {log_file} 失败: {str(e)}")
            
            # 清理报警图片和视频中较旧的文件
            alarm_dirs = [Path("JENS_data/alarm_images"), Path("JENS_data/alarm_videos")]
            for alarm_dir in alarm_dirs:
                if alarm_dir.exists():
                    # 删除30天前的报警资源
                    cutoff_date = datetime.now() - timedelta(days=30)
                    for file in alarm_dir.glob("*"):
                        try:
                            if file.is_file() and file.stat().st_mtime < cutoff_date.timestamp():
                                file.unlink()
                                logger.debug(f"已删除旧报警资源: {file}")
                        except Exception as e:
                            logger.error(f"删除报警资源 {file} 失败: {str(e)}")
            
            logger.info("磁盘清理完成")
            return True
            
        except Exception as e:
            logger.error(f"清理磁盘空间失败: {str(e)}")
            return False
    
    def _restart_system(self):
        """重启系统（仅在极端情况下使用）"""
        logger.warning("系统状态严重，准备重启...")
        
        # 记录重启日志
        try:
            restart_log_path = Path("JENS_data/logs/restarts.log")
            restart_log_path.parent.mkdir(exist_ok=True)
            
            with open(restart_log_path, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 系统状态严重，触发自动重启\n")
        except Exception as e:
            logger.error(f"写入重启日志失败: {str(e)}")
        
        # 执行重启命令
        try:
            if platform.system() == 'Windows':
                os.system('shutdown /r /t 60 /c "JENS系统自动重启"')
                logger.warning("已发出Windows重启命令，系统将在60秒后重启")
            else:
                os.system('sudo shutdown -r +1 "JENS系统自动重启"')
                logger.warning("已发出Linux/Mac重启命令，系统将在1分钟后重启")
            
            return True
            
        except Exception as e:
            logger.error(f"重启系统失败: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict:
        """
        获取系统信息
        
        返回:
            Dict: 系统信息字典
        """
        try:
            # 获取系统基本信息
            system_info = {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'platform_release': platform.release(),
                'hostname': platform.node(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(logical=False),  # 物理CPU核心数
                'cpu_count_logical': psutil.cpu_count(logical=True),  # 逻辑CPU核心数
                'memory_total': psutil.virtual_memory().total / (1024 * 1024),  # MB
                'disk_total': psutil.disk_usage('/').total / (1024 * 1024),  # MB
                'uptime': time.time() - psutil.boot_time()  # 系统运行时间(秒)
            }
            
            # 获取网络接口信息
            network_info = {}
            for interface_name, interface_addresses in psutil.net_if_addrs().items():
                for address in interface_addresses:
                    if address.family == 2:  # IPv4
                        if interface_name not in network_info:
                            network_info[interface_name] = {}
                        network_info[interface_name]['ipv4'] = address.address
                    elif address.family == 23:  # IPv6
                        if interface_name not in network_info:
                            network_info[interface_name] = {}
                        network_info[interface_name]['ipv6'] = address.address
            
            system_info['network'] = network_info
            
            return system_info
            
        except Exception as e:
            logger.error(f"获取系统信息失败: {str(e)}")
            return {
                'error': f"获取系统信息失败: {str(e)}"
            }
    
    def get_resource_history(self) -> Dict:
        """
        获取资源使用历史
        
        返回:
            Dict: 资源历史记录
        """
        return self.resource_history
    
    def get_process_info(self) -> List[Dict]:
        """
        获取进程信息
        
        返回:
            List[Dict]: 进程信息列表
        """
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'username': proc_info.get('username', ''),
                        'cpu_percent': proc_info['cpu_percent'],
                        'memory_percent': proc_info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # 按CPU使用率排序
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            return processes[:10]  # 返回前10个进程
            
        except Exception as e:
            logger.error(f"获取进程信息失败: {str(e)}")
            return []
    
    def set_monitoring_config(self, enable: bool = None, interval: int = None, 
                            log_interval: int = None):
        """设置监控配置"""
        if enable is not None:
            self.enable_monitoring = enable
        if interval is not None:
            self.monitoring_interval = max(10, interval)  # 最小10秒
        if log_interval is not None:
            self.log_interval = max(60, log_interval)  # 最小60秒
        
        # 更新配置
        self.monitor_config['enable'] = self.enable_monitoring
        self.monitor_config['interval'] = self.monitoring_interval
        self.monitor_config['log_interval'] = self.log_interval
        
        self.config.set('monitor', self.monitor_config)
        
        # 如果启用了监控但没有运行，则启动监控
        if self.enable_monitoring and not self.is_monitoring:
            self.start_monitoring()
        # 如果禁用了监控但正在运行，则停止监控
        elif not self.enable_monitoring and self.is_monitoring:
            self.stop_monitoring()
            
        logger.info(f"监控配置已更新: 启用={self.enable_monitoring}, 间隔={self.monitoring_interval}秒, 日志间隔={self.log_interval}秒")
    
    def set_threshold(self, cpu_warning: int = None, cpu_critical: int = None,
                    memory_warning: int = None, memory_critical: int = None,
                    disk_warning: int = None, disk_critical: int = None):
        """设置阈值配置"""
        if cpu_warning is not None:
            self.thresholds['cpu_warning'] = cpu_warning
        if cpu_critical is not None:
            self.thresholds['cpu_critical'] = cpu_critical
        if memory_warning is not None:
            self.thresholds['memory_warning'] = memory_warning
        if memory_critical is not None:
            self.thresholds['memory_critical'] = memory_critical
        if disk_warning is not None:
            self.thresholds['disk_warning'] = disk_warning
        if disk_critical is not None:
            self.thresholds['disk_critical'] = disk_critical
        
        # 更新配置
        self.monitor_config['thresholds'] = self.thresholds
        self.config.set('monitor', self.monitor_config)
        
        logger.info(f"阈值配置已更新: CPU警告={self.thresholds['cpu_warning']}%, CPU严重={self.thresholds['cpu_critical']}%, "
                   f"内存警告={self.thresholds['memory_warning']}%, 内存严重={self.thresholds['memory_critical']}%, "
                   f"磁盘警告={self.thresholds['disk_warning']}%, 磁盘严重={self.thresholds['disk_critical']}%")
    
    def set_recovery_config(self, enable: bool = None, max_cpu: int = None, max_memory: int = None,
                          min_free_space: int = None, recovery_actions: List[str] = None):
        """设置恢复配置"""
        if enable is not None:
            self.recovery_config['enable'] = enable
        if max_cpu is not None:
            self.recovery_config['max_cpu'] = max_cpu
        if max_memory is not None:
            self.recovery_config['max_memory'] = max_memory
        if min_free_space is not None:
            self.recovery_config['min_free_space'] = min_free_space
        if recovery_actions is not None:
            self.recovery_config['recovery_actions'] = recovery_actions
        
        # 更新配置
        self.monitor_config['recovery'] = self.recovery_config
        self.config.set('monitor', self.monitor_config)
        
        logger.info(f"恢复配置已更新: 启用={self.recovery_config['enable']}, 最大CPU={self.recovery_config['max_cpu']}%, "
                   f"最大内存={self.recovery_config['max_memory']}%, 最小可用空间={self.recovery_config['min_free_space']}MB, "
                   f"恢复动作={self.recovery_config['recovery_actions']}")


# 单例模式
_system_monitor_instance = None

def get_system_monitor() -> SystemMonitor:
    """
    获取系统监控器单例
    
    返回:
        SystemMonitor: 系统监控器实例
    """
    global _system_monitor_instance
    
    if _system_monitor_instance is None:
        _system_monitor_instance = SystemMonitor()
    
    return _system_monitor_instance


if __name__ == "__main__":
    # 测试代码
    monitor = get_system_monitor()
    
    # 获取系统状态
    status = monitor.check_system_status()
    print("系统状态:", status)
    
    # 获取系统信息
    system_info = monitor.get_system_info()
    print("系统信息:", system_info)
    
    # 获取进程信息
    processes = monitor.get_process_info()
    print("进程信息:", processes)
    
    # 启动监控
    # monitor.start_monitoring()
    # time.sleep(10)
    # monitor.stop_monitoring() 