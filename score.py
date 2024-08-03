import os

def load_labels(folder_path):
    '''
        加载标签文件中的类别（第一个值）
    '''
    labels = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                line = file.readline().strip()
                if line:
                    labels[filename] = line.split()[0]
    return labels

def find_unique_files(folder_a, folder_b):
    # 获取两个文件夹中的文件列表
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))
    
    # 找出A文件夹中有但B文件夹中没有的文件
    unique_files = files_a - files_b
    
    return unique_files

def del_unique_file(unique_files, true_labels):
    '''
        删除true_labels中多出的键值对
    '''
    for unique_file in unique_files:
        if unique_file in true_labels:
            del true_labels[unique_file]

    return true_labels

def compare_labels(true_labels, pred_labels):
    match_count = 0
    for key in true_labels:
        if key in pred_labels and true_labels[key] == pred_labels[key]:
            match_count += 1
    return match_count

if __name__ == "__main__":
    # 设置文件夹路径
    true_labels_folder = r'C:\Users\Lazzy\Desktop\Project\breastUltrasound\database\test_A\乳腺分类测试集A\A\2类\labels'
    pred_labels_folder = r'C:\Users\Lazzy\Desktop\Project\breastUltrasound\runs\detect\predict4\labels'

    # 加载标签
    true_labels = load_labels(true_labels_folder)
    pred_labels = load_labels(pred_labels_folder)

    total = len(true_labels)
    print(f"total = {total}")

    # 获取真实标签文件夹中有但预测标签文件夹中没有的文件
    unique_files = list(find_unique_files(true_labels_folder, pred_labels_folder))
    # 删除真实标签字典中没有的
    del_true_labels = del_unique_file(unique_files, true_labels)
    print(f"deled = {len(del_true_labels)}")
    # 预测正确的
    match_count = compare_labels(del_true_labels, pred_labels)

    print(match_count)

    Accuracy = match_count / total *100

    print(f"Accuracy = {Accuracy}")