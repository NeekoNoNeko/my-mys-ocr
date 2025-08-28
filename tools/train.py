from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r'yolo11n.pt')
    # model.train(resume=True)
    model.train(data=r'train.yaml',
                epochs=600,
                patience=20,  # 早停耐心值：50个epoch没有改善就停止
                save_period=10,  # 每10个epoch保存一次模型
                val=True,  # 启用验证
                plots=True,  # 生成训练图表
                save=True,  # 保存模型
                device='0')  # 自动选择设备
