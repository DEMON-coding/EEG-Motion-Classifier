本目录为开发版，如需查看稳定版本，请访问原目录 `EEG-Motion-Classifier`

---

* 添加了 `PredictTimely.py` 支持调用 `models_dict/` 下的模型进行实时预测，使用时请注意输入输出设备端口的配置(示例为商家的USB口输入与网址段传值)
* 添加了 `GetRaw.py` 支持直接从设备端口读取EEG原始信号，结果以`.txt`格式保存在 `data_dev/` 目录下，格式相同，可以用 `DataSet_dev.py` 整理为`.csv`文件，并用 `Train_dev.py` 训练。
* 在 `GetRaw.py` 中添加了注意力判断，在注意力大于阈值时才进行信号的读取， `GetFeature.py` 专用于注意力的测量，确定阈值。 

---

## 调试中

* `test_dev.py` 用以测试实时预测，独立了网址端用以调试，准备加入多线程与滑窗采集等减少实时预测的误差