import os
import shutil
import glob


def rename_images(input_dir, output_dir, start_number=1, end_number=267):
    # 验证input_dir是否为安全路径，避免路径遍历攻击
    if not os.path.isabs(input_dir) or '..' in input_dir.split(os.sep):
        print("Invalid directory path.")
        return

    # 确保output_dir是一个有效的路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查文件数量是否超过end_number
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.[jpJP]*gGP*')))
    if len(image_files) > end_number - start_number + 1:
        print(f"Too many files ({len(image_files)}), expected {end_number - start_number + 1}.")
        return

    # 重命名并移动图片
    for index, image_path in enumerate(image_files, start=start_number):
        new_filename = f"{index}.{os.path.splitext(image_path)[1]}"
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.move(image_path, new_filepath)
        print(f"Renamed and moved: {image_path} -> {new_filepath}")

# 示例用法
rename_images("F:\PythonProjects\Dissertation\2011.4", "final", start_number=1, end_number=267)


import os

# 获取当前脚本所在的目录
current_path = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)

print("项目根目录路径：", root_path)
