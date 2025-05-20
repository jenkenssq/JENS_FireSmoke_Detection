# JENS火灾烟雾报警系统

基于PySide6和YOLOv5的火灾烟雾实时检测与报警系统，通过计算机视觉技术识别视频流中的火灾和烟雾，并进行及时报警。系统适用于工厂、仓库、办公楼等场所的安全监控。（目前只有视频能识别成功，图片不能识别TUT）

## 功能特点

- 实时检测视频中的火灾和烟雾
- 多级报警机制（低、中、高级别）
- 支持本地摄像头和IP摄像头接入
- 多摄像头同时监控
- 可定义检测区域，提高检测精度
- 历史报警记录查询和统计
- 报警画面保存与回放
- 报警事件自动处理流程
- 支持数据分析和报表生成
- 基于MySQL的数据持久化存储

## 系统要求

### 硬件要求
- **处理器**：Intel i5 8代或更高/AMD Ryzen 5 或更高
- **内存**：至少8GB RAM，推荐16GB
- **存储**：至少20GB可用空间
- **显卡**：集成显卡(基础模式)或NVIDIA GTX 1050+(高性能模式)
- **网络**：支持有线/无线网络连接

### 软件要求
- **操作系统**：Windows 10/11（64位）或 Linux (Ubuntu 18.04+)
- **Python**：3.8+
- **MySQL**：5.7+ 数据库
- **CUDA**：10.2+(可选，用于GPU加速)

## 安装步骤

### 1. 克隆代码库

```bash
git clone https://github.com/jenkenssq/JENS_FireSmoke_Detection.git
cd JENS_FireSmoke_Detection
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备YOLOv5模型

使用内置的模型下载脚本：

```bash
python JENS_download_models.py
```

或手动下载预训练模型到 `JENS_models` 目录

### 4. 配置MySQL数据库

#### 4.1 使用MySQL命令行工具

```bash
# 登录MySQL
mysql -u root -p

# 运行初始化脚本
source JENS_utils/init_database.sql
```

#### 4.2 使用图形化工具(如MySQL Workbench)

1. 打开MySQL Workbench并连接到您的MySQL服务器
2. 创建一个新的查询标签
3. 将`JENS_utils/init_database.sql`文件内容复制到查询窗口
4. 执行脚本

#### 4.3 修改数据库连接配置

在首次运行程序时，系统将创建`config.json`文件。您可以手动编辑此文件修改数据库连接信息：

```json
{
  "database": {
    "host": "localhost",
    "port": 3306,
    "user": "your_username",
    "password": "your_password",
    "database": "jens_fire_smoke",
    "charset": "utf8mb4"
  }
}
```

## 运行系统

### 基本运行

```bash
python JENS_main.py
```

### 高级选项

```bash
# 使用自定义配置文件
python JENS_main.py --config custom_config.json

# 无界面模式
python JENS_main.py --no-gui
```

## 系统配置

### 配置文件

系统配置存储在`config.json`文件中，包括：

- 数据库连接信息
- 检测模型参数
- 摄像头设置
- 报警阈值
- 存储路径配置

### 用户账户

系统初始管理员账户：
- 用户名：admin
- 密码：admin123

首次登录后请立即修改密码。

## 目录结构

```
JENS_FireSmoke_Detection/
├── JENS_main.py            # 主程序入口
├── JENS_detector.py        # YOLOv5检测器类
├── JENS_camera.py          # 摄像头和视频处理类
├── JENS_camera_pool.py     # 摄像头池管理
├── JENS_alarm.py           # 报警系统类
├── JENS_monitor.py         # 系统监控模块
├── JENS_gui/               # GUI相关模块
├── JENS_models/            # YOLOv5模型存放目录
├── JENS_utils/             # 工具模块
│   ├── JENS_config.py      # 配置文件处理
│   ├── JENS_logger.py      # 日志系统
│   ├── JENS_database.py    # MySQL数据库操作类
│   ├── JENS_analytics.py   # 数据分析工具
│   └── init_database.sql   # 数据库初始化脚本
├── JENS_docs/              # 项目文档
│   ├── user_manual/        # 用户手册
│   └── deployment/         # 部署指南
├── JENS_assets/            # 图像、声音等资源
├── JENS_storage/           # 报警记录存储目录
│   ├── images/             # 报警截图
│   └── videos/             # 报警视频片段
├── JENS_data/              # 数据存储目录
├── download_models.py      # 模型下载工具
├── logs/                   # 日志目录
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明 
```

## 文档

- [用户手册](JENS_docs/user_manual/JENS_user_manual.md) - 详细的系统使用指南
- [部署指南](JENS_docs/deployment/JENS_deployment_guide.md) - 系统部署和维护说明

## 开发团队

- 项目负责人：jenkenssq
- 联系邮箱：jenkens@qq.com
- Github：https://github.com/jenkenssq

## 许可协议

本项目采用MIT许可协议。详见LICENSE文件。 
