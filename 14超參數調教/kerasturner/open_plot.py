import os
import json
import hiplot as hip

# 定義資料夾路徑
rootdir = 'C:/Users/User/Desktop/self_learning_code/超參數調教/kerasturner/my_dir/test1'
print(os.listdir(rootdir))  # 列印目錄內容，檢查是否有 trial.json 檔案

vis_data = []

# 掃描目錄內每個檔案
for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("trial.json"):
            with open(subdirs + '/' + file, 'r') as json_file:
                data = json_file.read()
                vis_data.append(json.loads(data))

print(vis_data)  # 檢查資料是否正確加載
#import hiplot as hip
# 建立字典，包含參數組合與測試結果
data = [{
            'units': vis_data[idx]['hyperparameters']['values']['units'],
            'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'],
            'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],
            'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'],
            'accuracy': vis_data[idx]['metrics']['metrics']['accuracy']['observations'][0]['value'],
            'val_accuracy': vis_data[idx]['metrics']['metrics']['val_accuracy']['observations'][0]['value']
        } for idx in range(len(vis_data))]

# 顯示交互式圖表
#hiplot圖表變成網頁在資料夾內
output_path = 'C:/Users/User/Desktop/self_learning_code/超參數調教/kerasturner/my_dir/test1/experiment_results.html'
experiment = hip.Experiment.from_iterable(data)
html_content = experiment.to_html()

# 儲存為HTML檔案
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)


