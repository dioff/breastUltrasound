from ultralytics import YOLO

weights_path = 'yolov8s.pt' 
model = YOLO(weights_path, task='train')
ksplit = 5
# 从文本文件中加载内容并存储到一个列表中
ds_yamls = []
with open('data/file_paths.txt', 'r') as f:
    for line in f:
        # 去除每行末尾的换行符
        line = line.strip()
        ds_yamls.append(line)

# 打印加载的文件路径列表
print(ds_yamls)


results = {}
for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model.train(data=dataset_yaml, batch=6, epochs=2, imgsz=1280, device=0, workers=8, single_cls=False, ) 
