from ultralytics import YOLO



if __name__ == "__main__":

    # NB: 我也不知道为什么，我以开是写的是model=YOLO("yolov8s.pt")，但是准确率才96.8
    # 但是我用下面这条后，就变成了99.2，好像是因为这个是吧权重载入，具体的我也没搞懂
    # 看了官网以后知道的答案，只用pt或者yml的话就是不使用预训练模型
    model = YOLO("yolov8s.yaml").load("yolov8s.pt")

    # model = YOLO("yolov8-seg.yaml").load("yolov8-seg.pt")

    dataPath = r'C:\Users\Lazzy\Desktop\Project\breastUltrasound\2\2.yaml'

    model.train(data=dataPath, epochs=20, save = True, batch=64)
    
    metrics = model.val()