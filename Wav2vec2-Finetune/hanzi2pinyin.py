from pypinyin import pinyin
import os
from tqdm import tqdm

def add_suffix_to_txt_files(file_name):
    base_name = os.path.basename(file_name)
    new_name = os.path.splitext(base_name)[0] + "_pipyin.txt"
    new_path = os.path.join(os.path.dirname(file_name), new_name)
    return new_path

def get_all_txt_files(root_folder):
    if not os.path.exists(root_folder):
        print(f"Thư mục '{root_folder}' không tồn tại.")
        return []
    txt_files = [os.path.join(root, file) for root, dirs, files in os.walk(root_folder) for file in files if file.endswith('.txt')]

    return txt_files

def convert_pinyin(input_folder):
    txt_files = get_all_txt_files(input_folder)
    print(len(txt_files))
    total_files = len(txt_files)
    files_processed = 0  # Biến đếm số lượng tệp tin đã xử lý
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            pinyin_result = pinyin(content)
            result = ' '.join([item[0] for item in pinyin_result])
            
            new_path = add_suffix_to_txt_files(file_path)
            
            with open(new_path, 'w', encoding='utf-8') as new_file:
                new_file.write(result)
            files_processed += 1
            print(f"Processed file {files_processed}/{total_files}: {file_path} -> {new_path}")
                
    print("Conversion completed.")


# Sử dụng hàm để chuyển đổi và in thông tin tiến trình
input_folder = input("input path you want to convert :))))): ")
convert_pinyin(str(input_folder))
