import json
import os

default_config_path = "configs/local_default_config.json"
smoke_config_path = "configs/smoke_test_config.json"

# 读取官方默认配置
with open(default_config_path, "r") as f:
    config = json.load(f)

# 暴力覆写我们的冒烟测试参数
config["pipeline"] = "supervised"
config["task"] = "MotionSourceRecognition"
config["model"] = "resnet18"
config["batch_size"] = 32
config["epochs"] = 2

# 兜底：强行修正数据读取路径（覆盖常见的几种键名）
data_path = "/root/autodl-tmp/CSI_MultiTask_Project/data"
for key in ["data_dir", "dataset_dir", "data_path", "root_path"]:
    if key in config:
        config[key] = data_path
# 如果全都没匹配上，强行写入最常见的 data_dir
if "data_dir" not in config and "dataset_dir" not in config:
    config["data_dir"] = data_path

# 另存为新的配置文件
with open(smoke_config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"✅ 冒烟配置已生成！(保存至 {smoke_config_path})")
