# keras tuner 超參數調校
## 步驟1：載入 MNIST 手寫阿拉伯數字資料
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# 載入 MNIST 手寫阿拉伯數字資料
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 訓練/測試資料的 X/y 維度
print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)
## 步驟2：資料清理，此步驟無需進行
## 步驟3：進行特徵工程，將特徵縮放成(0, 1)之間
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0
## 步驟4：資料分割，此步驟無需進行，載入MNIST資料時，已經切割好了

## 步驟5：建立模型結構

## 步驟6：結合訓練資料及模型，進行模型訓練

## 步驟7：評分(Score Model)
#pip install keras-tuner
import kerastuner as kt

# 建立模型
def model_builder(hp):
    # 學習率(learning rate)選項： 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values = [0.01, 0.001, 0.0001]) 
    # 第一層Dense輸出選項： 32、64、...、512
    hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(hp_units, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
# 調校設定， Hyperband：針對所有參數組合進行測試
tuner = kt.Hyperband(model_builder,              # 模型定義
                     objective = 'val_accuracy', # 目標函數
                     max_epochs = 5,             # 最大執行週期
                     factor = 2,                 # 執行週期數的遞減因子
                     directory = 'my_dir',       # 存檔目錄
                     project_name = 'test2')     # 專案名稱
# 參數調校
import IPython

# 每個參數組合測完後，清除輸出顯示
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)


# 調校執行
tuner.search(x_train_norm, y_train, epochs = 5, 
             validation_data = (x_test_norm, y_test), # 驗證資料
             callbacks = [ClearTrainingOutput()])     # 執行每個參數組合後回呼

# 顯示最佳參數值
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"最佳參數值\n第一層Dense輸出：{best_hps.get('units')}\n", 
        f"學習率：{best_hps.get('learning_rate')}")
tuner.search_space_summary()
tuner.results_summary()
tuner.get_best_models(num_models=1)[0]
## Hiplot 視覺化
# 解析 Keras Tuner 測試的日誌檔
import os
import json

vis_data = []
# 掃描目錄內每一個檔案
rootdir = 'my_dir/test1'
for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("trial.json"):
            with open(subdirs + '/' + file, 'r') as json_file:
                data = json_file.read()
            vis_data.append(json.loads(data))
# 顯示參數組合與測試結果
import hiplot as hip

# 建立字典，含參數組合與測試結果
data = [{'units': vis_data[idx]['hyperparameters']['values']['units'], 
         'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
         'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],  
         'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'], 
         'accuracy': vis_data[idx]['metrics']['metrics']['accuracy']['observations'][0]['value'],
         'val_accuracy': vis_data[idx]['metrics']['metrics']['val_accuracy']['observations'][0]['value']} 
        for idx in range(len(vis_data))]

# 顯示
hip.Experiment.from_iterable(data).display()
print(f"實際最大試驗數: {len(tuner.oracle.trials)}")

