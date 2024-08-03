from ultralytics import YOLO
 

def Prediction(imagePath):
    '''
        该函数用于预测
        imagePath：图片路径
    '''

    results = model.predict(imagePath, save=True, save_txt=True)
    for r in results[0]:
        if r.boxes.cls.item()==0.0:
            print('无')
        elif r.boxes.cls.item()==1.0:
            print('class 2')


if __name__ == "__main__":
    # 数据集路径
    image_dir = r"C:\Users\Lazzy\Desktop\Project\breastUltrasound\database\test_A\乳腺分类测试集A\A\2类\images"
    # 图片路径
    image_path = r'database/test_A/乳腺分类测试集A/A/3类/images/0097.jpg'
    # 模型路径
    modelPath = r'C:\Users\Lazzy\Desktop\Project\breastUltrasound\runs\detect\train\weights\best.pt'

    # 加载模型
    model = YOLO(modelPath)
    Prediction(image_dir)

 