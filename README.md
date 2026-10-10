# 🌊 常大河道水面垃圾巡检系统 V2.0

基于 **YOLOv8 + PyQt6** 的水面垃圾检测工具，支持摄像头实时监控、图片文件夹批量分析、垃圾数量可视化。

## 下载 Windows 程序

[查看 V2.0 Release](https://github.com/lichuanjie31-prog/Computer-Vision-River-Cleanup-CCZU/releases/tag/v2.0) · [下载已发布的 _V2.0.exe](https://github.com/lichuanjie31-prog/Computer-Vision-River-Cleanup-CCZU/releases/download/v2.0/_V2.0.exe)

注意：上述 Windows 程序是**之前发布**的 V2.0 版本，不包含本次 Pull Request 的代码修复。合并代码后需要重新打包发布才能获得修复。

## 从源码运行

建议 Python 3.11 或 3.12，Windows PowerShell：

~~~powershell
git clone https://github.com/lichuanjie31-prog/Computer-Vision-River-Cleanup-CCZU.git
cd Computer-Vision-River-Cleanup-CCZU
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

如需 NVIDIA 显卡加速，请先根据 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 配置适配显卡的 PyTorch 版本。

**模型权重不在 Git 目录中。** 请在 [Release 附件](https://github.com/lichuanjie31-prog/Computer-Vision-River-Cleanup-CCZU/releases/tag/v2.0) 下载 `best.pt`，放在项目根目录的 `weights/best.pt`（也支持根目录的 `best.pt`）。

~~~powershell
python river_app_final.py
~~~

## 功能与修复

- 模式 A：本地摄像头实时检测与数量曲线。
- 模式 B：批量识别 JPG/JPEG/PNG 图片，输出到所选文件夹的 `Results_Output_V2/`。
- 批量推理现运行在 QThread 工作线程，避免主窗口因长批处理长期无响应。
- “停止任务”会发送取消请求；当前单张图片正在进行的模型推理结束后才会停止。
- 优化摄像头释放、窗口关闭、缺失模型提示及输出异常处理。
- 图片使用 Unicode 路径兼容的读写方式，可处理中文文件路径。
- 摄像头模式仍在 GUI 线程内推理，低性能设备可能短暂卡顿。

## 训练数据

`data.yaml` 已改为相对配置，预期在仓库根目录建立：

~~~text
dataset/
  train/images/
  valid/images/
  test/images/
~~~

数据集未随仓库提供。历史数据来源：[Roboflow Trash in River v2](https://universe.roboflow.com/try-uqrjj/trash-in-river/dataset/2)，标注 CC BY 4.0，请遵守来源许可。

原 README 中的精度提升 30% 和单帧低于 5ms 暂无仓库内可复现实验记录，因此不作为已验证指标。训练和实际推理还需要真实硬件与权重文件验证。

## 检查

~~~powershell
python -m py_compile river_app_final.py
python -m unittest discover -s tests -v
~~~

GitHub Actions 在 PR 上运行上述基础静态检查。通过并不能代替 Windows、显卡、摄像头及模型权重的真实端到端验收。

## 团队

常州大学软件252 · 李传杰（组长）；组员：沈永权、褚恒智、蔡宜辰、伏丽学、梁大娟、徐可欣、林星言、潘彤、宋雯萱。

感谢常州大学阿里云学院李宁老师指导。源码使用 MIT License；第三方模型与数据集遵循各自许可。
