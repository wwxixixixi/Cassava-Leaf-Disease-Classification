# Cassava-Leaf-Disease-Classification
kaggle competiton 

## Dataloader
- 随机获取指定batch_size的数据（包括test时一次取出所有图片）
- 取出的数据格式兼容model所需的tensor
- 图片预处理函数可重写

## 数据增强
- 读取现有图片进行翻转，旋转等变换
- 生成新的数据集

## Model
- 继承nn.module重写__init__和forward

## Trainer
- 定义loss
- 训练主循环及超参数调节

## Test
- 将模型上传到jupyter
- 读取测试集图片进行推理
- 将结果整理成submission.csv的形式
- 获得得分
