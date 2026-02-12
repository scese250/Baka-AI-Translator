import os
import shutil
import tempfile
from .archives import extract_archive, natural_sort_key

class FileHandler:
    def __init__(self):
        self.file_paths = []
        self.archive_info = []
    
    def prepare_files(self, file_paths: list[str], extend: bool = False):
        all_image_paths = []
        if not extend:
            for archive in self.archive_info:
                temp_dir = archive['temp_dir']
                if os.path.exists(temp_dir): 
                    shutil.rmtree(temp_dir)  
            self.archive_info = []
        
        # Sort input paths using natural sort for proper numeric ordering (1, 2, 10 not 1, 10, 2)
        file_paths = sorted(file_paths, key=natural_sort_key)
        
        for path in file_paths:
            if path.lower().endswith(('.cbr', '.cbz', '.cbt', '.cb7', 
                                      '.zip', '.rar', '.7z', '.tar',
                                      '.pdf', '.epub')):
                print('Extracting archive:', path)
                archive_dir = os.path.dirname(path)
                temp_dir = tempfile.mkdtemp(dir=archive_dir)
                
                extracted_files = extract_archive(path, temp_dir)
                image_paths = [f for f in extracted_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
                
                all_image_paths.extend(image_paths)               
                self.archive_info.append({
                    'archive_path': path,
                    'extracted_images': image_paths,
                    'temp_dir': temp_dir
                })
            elif os.path.isdir(path):
                print('Scanning directory:', path)
                for root, dirs, files in os.walk(path):
                    # Sort dirs and files using natural sort for proper numeric ordering
                    dirs.sort(key=natural_sort_key)
                    files.sort(key=natural_sort_key)
                    
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                            full_path = os.path.join(root, file)
                            all_image_paths.append(full_path)
            else:
                all_image_paths.append(path)
        
        self.file_paths = self.file_paths + all_image_paths if extend else all_image_paths
        return all_image_paths
