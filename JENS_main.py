#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JENS火灾烟雾报警系统主程序
"""

import os
import sys
import time
import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from JENS_gui.JENS_main_window import MainWindow
from JENS_utils.JENS_config import Config
from JENS_utils.JENS_logger import get_logger, JENSLogger
from JENS_camera import get_camera_manager
from JENS_detector import get_detector, get_async_detector
from JENS_alarm import get_alarm_manager

# 移除通知模块引用
NOTIFICATION_AVAILABLE = False

# 导入其他模块
try:
    from JENS_monitor import get_system_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from JENS_camera_pool import get_camera_pool
    CAMERA_POOL_AVAILABLE = True
except ImportError:
    CAMERA_POOL_AVAILABLE = False

try:
    from JENS_utils.JENS_analytics import get_data_analytics
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# 初始化日志
logger_manager = JENSLogger()  # 初始化日志系统
logger = get_logger("JENS_main")

def check_dependencies():
    """检查依赖项"""
    logger.info("正在检查依赖项...")
    
    missing_dependencies = []
    
    # 检查PySide6
    try:
        import PySide6
        logger.info(f"检测到PySide6 {PySide6.__version__}")
    except ImportError:
        missing_dependencies.append("PySide6")
    
    # 检查PyTorch
    try:
        import torch
        logger.info(f"检测到PyTorch {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            logger.info(f"GPU加速已启用：{torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU加速未启用，将使用CPU进行推理")
    except ImportError:
        missing_dependencies.append("torch")
    
    # 检查OpenCV
    try:
        import cv2
        logger.info(f"检测到OpenCV {cv2.__version__}")
    except ImportError:
        missing_dependencies.append("opencv-python")
    
    # 检查NumPy
    try:
        import numpy
        logger.info(f"检测到NumPy {numpy.__version__}")
    except ImportError:
        missing_dependencies.append("numpy")
    
    # 检查其他常用库
    for lib_name in ["matplotlib", "pandas", "seaborn", "psutil"]:
        try:
            lib = __import__(lib_name)
            if hasattr(lib, "__version__"):
                logger.info(f"检测到{lib_name} {lib.__version__}")
            else:
                logger.info(f"检测到{lib_name}")
        except ImportError:
            missing_dependencies.append(lib_name)
    
    if missing_dependencies:
        logger.warning(f"以下依赖项未安装：{', '.join(missing_dependencies)}")
        logger.info("请使用以下命令安装缺失的依赖项：")
        logger.info(f"pip install {' '.join(missing_dependencies)}")
        return False
    else:
        logger.info("所有依赖项检查通过")
        return True

def check_model_files():
    """检查模型文件"""
    logger.info("检查模型文件...")
    
    models_dir = Path("JENS_models")
    if not models_dir.exists():
        logger.warning(f"模型目录不存在: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"已创建模型目录: {models_dir}")
    
    # 检查默认模型文件
    default_model = models_dir / "yolov5s_best.pt"  # 使用火灾检测专用模型
    if not default_model.exists():
        logger.warning(f"默认模型文件不存在: {default_model}")
        logger.info("请下载模型文件或运行download_models.py脚本")
        return False
    
    logger.info("模型文件检查通过")
    return True

def initialize_components(config):
    """初始化各个组件"""
    logger.info("初始化系统组件...")
    
    # 初始化摄像头管理器
    camera_manager = get_camera_manager()
    if not camera_manager:
        logger.error("初始化摄像头管理器失败")
        return False
    
    # 初始化检测器
    model_path = "D:\\监控系统\\JENS_FireSmoke_Detection\\JENS_models\\yolov5s_best.pt"  # 指定绝对路径
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    # 创建检测器
    try:
        detector = get_detector(model_path)
        if not detector:
            logger.error(f"初始化检测器失败，无法加载模型: {model_path}")
            logger.error(f"请检查JENS_detector.log获取详细错误信息")
            return False
            
        # 配置检测器
        detector_config = config.get_config('detector')
        if detector_config:
            # 设置置信度阈值
            conf_thres = detector_config.get('confidence_threshold')
            if conf_thres:
                detector.set_conf_thres(float(conf_thres))
                logger.info(f"已设置检测器置信度阈值: {conf_thres}")
            
            # 设置检测频率
            detection_freq = detector_config.get('detection_frequency')
            if detection_freq:
                detector.set_detection_freq(int(detection_freq))
                logger.info(f"已设置检测频率: {detection_freq}")
            
            # 启用增强模式
            enable_enhanced = detector_config.get('enable_enhanced', True)
            detector.enable_enhanced_mode(enable_enhanced)
            logger.info(f"增强检测模式: {'已启用' if enable_enhanced else '已禁用'}")
            
    except Exception as e:
        logger.error(f"初始化检测器时发生错误: {str(e)}")
        return False
    
    # 初始化报警管理器
    alarm_manager = get_alarm_manager()
    if not alarm_manager:
        logger.error("初始化报警管理器失败")
        return False
    
    # 初始化新组件 (移除通知管理器初始化)
    
    if MONITORING_AVAILABLE:
        system_monitor = get_system_monitor()
        # 启动系统监控
        system_monitor.start_monitoring()
        logger.info("系统监控已启动")
    
    if CAMERA_POOL_AVAILABLE:
        camera_pool = get_camera_pool()
        logger.info("摄像头池已初始化")
    
    if ANALYTICS_AVAILABLE:
        data_analytics = get_data_analytics()
        logger.info("数据分析器已初始化")
    
    logger.info("所有组件初始化完成")
    return True

def start_gui(config):
    """启动GUI界面"""
    logger.info("正在启动GUI界面...")
    
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("JENS火灾烟雾报警系统")
    app.setOrganizationName("JENS")
    
    # 设置应用程序图标
    icon_path = os.path.join("JENS_gui", "logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        logger.info(f"已设置应用程序图标: {icon_path}")
    else:
        logger.warning(f"应用程序图标文件不存在: {icon_path}")
    
    # 创建主窗口
    main_window = MainWindow(config)
    main_window.show()
    
    logger.info("GUI界面已启动")
    return app.exec()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="JENS火灾烟雾报警系统")
    parser.add_argument("--no-gui", action="store_true", help="不启动GUI界面")
    parser.add_argument("--config", type=str, help="指定配置文件路径")
    args = parser.parse_args()
    
    logger.info("JENS火灾烟雾报警系统正在启动...")
    
    # 加载配置
    config = Config()  # Config类是单例模式
    
    # 检查依赖项和模型文件
    if not check_dependencies() or not check_model_files():
        logger.warning("依赖项或模型文件检查未通过，系统可能无法正常工作")
    
    # 初始化组件
    if not initialize_components(config):
        logger.error("初始化组件失败，系统无法启动")
        return 1
    
    # 启动GUI或无界面模式
    if not args.no_gui:
        return start_gui(config)
    else:
        logger.info("系统以无界面模式启动")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，系统正在关闭...")
        
        # 清理资源
        if MONITORING_AVAILABLE:
            get_system_monitor().stop_monitoring()
        
        if CAMERA_POOL_AVAILABLE:
            get_camera_pool().stop_all()
        
        logger.info("系统已关闭")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 