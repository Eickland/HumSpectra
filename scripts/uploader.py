import os
import requests

# Настройки
API_URL = "http://mass-spectrim-webinterface.ru/api/upload"
API_KEY = "3fjeiu289fh8238h84fh781290"
LOCAL_DATA_DIR = "./processed_results" # Папка, где лежат ваши графики

def upload_files():
    if not os.path.exists(LOCAL_DATA_DIR):
        print(f"Ошибка: Папка {LOCAL_DATA_DIR} не найдена")
        return

    files_to_upload = []
    
    # Рекурсивный обход всех подпапок
    for root, dirs, files in os.walk(LOCAL_DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.xlsx')):
                # Полный путь к файлу
                full_path = os.path.join(root, file)
                # Для обратной совместимости: если файл в корневой папке, сохраняем только имя
                # Если во вложенной папке, сохраняем относительный путь
                if root == LOCAL_DATA_DIR:
                    # Плоская структура (как в старом коде)
                    file_identifier = file
                else:
                    # Вложенная структура - сохраняем путь относительно LOCAL_DATA_DIR
                    relative_path = os.path.relpath(full_path, LOCAL_DATA_DIR)
                    file_identifier = relative_path
                
                files_to_upload.append((full_path, file_identifier))
    
    print(f"Найдено файлов для загрузки: {len(files_to_upload)}")

    for file_path, file_identifier in files_to_upload:
        with open(file_path, 'rb') as f:
            # Используем file_identifier как имя файла при загрузке
            # Это может быть просто имя или относительный путь (например, "subfolder/file.png")
            files = {'file': (file_identifier, f)}
            headers = {'X-API-Key': API_KEY}
            
            try:
                response = requests.post(API_URL, headers=headers, files=files)
                if response.status_code == 200:
                    print(f"✅ Успешно: {file_identifier}")
                else:
                    print(f"❌ Ошибка {file_identifier}: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❗ Ошибка соединения при отправке {file_identifier}: {e}")

if __name__ == "__main__":
    upload_files()