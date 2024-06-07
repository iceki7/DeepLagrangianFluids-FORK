import os

#COPY
def is_folder_empty(folder_path):
    return not os.listdir(folder_path)
 
def search_empty_folders(root_path):
    for folder_name, subfolders, file_names in os.walk(root_path):
        if is_folder_empty(folder_name):
            print(f"Empty folder found: {folder_name}")
 
# 使用示例
root_path = r'D:\CODE\MCVSPH-FORK\c'  # 替换为你的目标目录
root_path = r'/w/cconv-dataset/mcvsph-dataset/csm'  # 替换为你的目标目录
search_empty_folders(root_path)