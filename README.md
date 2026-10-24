深度学习卷积神经网络_如何使用南瓜霜霉病数据集 建立基于深度学习Yolo南瓜病害检测
@[toc]
以下文字及代码仅供参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de8914c09bd64fa3a58fc4167a45b66a.png)
基于深度学习YOLO（You Only Look Once）目标检测算法，使用南瓜霜霉病数据集建立一个南瓜病害检测系统，

从数据准备、模型选择与训练到推理的全流程进行详细说明，并附上代码示例。

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e90a870091a480e8e923677fe300c6a.png)


### **1. 数据准备**
南瓜霜霉病数据集通常包括标注了南瓜叶片及其病害区域的图片。如果你已经有标注好的数据集，可以直接跳过这一步。如果没有，需要先对数据进行标注。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/55e0b4dfbca44c46b06e8292db3ca186.png)

#### **1.1 数据格式转换**
YOLO要求输入数据为特定格式：
- 图片文件夹：`images/`
- 标注文件夹：`labels/`
  - 每个图片对应一个 `.txt` 文件，包含目标框信息。
  - 格式：`class x_center y_center width height`（归一化值）

可以使用工具如 [LabelImg](https://github.com/tzutalin/labelImg) 或 [CVAT](https://github.com/openvinotoolkit/cvat) 对数据集进行标注。

#### **1.2 数据划分**
将数据集划分为训练集、验证集和测试集，例如：
- 训练集：80%
- 验证集：10%
- 测试集：10%

创建 `train.txt` 和 `val.txt` 文件，分别列出训练和验证集的图片路径。

---

### **2. 环境搭建**
确保安装了必要的依赖库：
```bash
pip install torch torchvision opencv-python matplotlib
```

我们可以使用 Ultralytics 提供的 YOLOv5 实现，它简单易用且功能强大。

克隆 YOLOv5 仓库：
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

### **3. 数据配置**
在 YOLOv5 中，创建一个 YAML 文件来定义数据集路径和类别。

创建 `pumpkin.yaml` 文件：
```yaml
# pumpkin.yaml
train: /path/to/train/images
val: /path/to/val/images

# 类别数量
nc: 1

# 类别名称
names: ['frost_mold']
```

---

### **4. 模型训练**
使用 YOLOv5 进行模型训练。

#### **4.1 配置超参数**
默认情况下，YOLOv5 提供了一些预训练权重（如 `yolov5s.pt`），可以选择适合的模型大小（s/m/l/x）。

运行训练命令：
```bash
python train.py --img 640 --batch 16 --epochs 50 --data pumpkin.yaml --weights yolov5s.pt
```

参数说明：
- `--img 640`：输入图像大小。
- `--batch 16`：批量大小。
- `--epochs 50`：训练轮数。
- `--data pumpkin.yaml`：数据配置文件。
- `--weights yolov5s.pt`：预训练权重。

训练完成后，最佳模型会保存在 `runs/train/exp/weights/best.pt`。

---

### **5. 模型推理**
使用训练好的模型对新图片进行预测。

#### **5.1 推理代码**
```python
from PIL import Image
import cv2
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# 加载测试图片
img_path = 'test_image.jpg'
img = cv2.imread(img_path)

# 推理
results = model(img)

# 显示结果
results.show()  # 可视化结果
results.save()  # 保存结果到文件夹
```

#### **5.2 批量推理**
如果需要对多个图片进行推理，可以使用循环：
```python
import os

# 测试图片文件夹
test_dir = 'test_images/'
output_dir = 'output/'

os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path)
    
    # 推理
    results = model(img)
    
    # 保存结果
    output_path = os.path.join(output_dir, img_name)
    results.save(output_path)
```

---

### **6. 性能评估**
可以使用验证集评估模型性能：
```bash
python val.py --data pumpkin.yaml --weights runs/train/exp/weights/best.pt
```

输出指标包括 mAP（mean Average Precision）、Precision 和 Recall。


仅供参考。我的同学。
