import subprocess

PYTHON_BIN = "/home/bio/anaconda3/envs/detectron2/bin/python"
TRAIN_SCRIPT = "/home/bio/Tanvir_Saikat/few-shot-object-detection/tools/train_net.py"

configs = [
    "configs/PascalVOC-detection/split1/faster_rcnn_R_101_DoG_FPN_V2/base1.yaml",
    "configs/PascalVOC-detection/split1/faster_rcnn_R_101_DoG_FPN_V2/1shot.yaml",
    "configs/PascalVOC-detection/split1/faster_rcnn_R_101_DoG_FPN_V2/3shot.yaml",
    "configs/PascalVOC-detection/split1/faster_rcnn_R_101_DoG_FPN_V2/5shot.yaml",
    "configs/PascalVOC-detection/split3/faster_rcnn_R_101_DoG_FPN_V2/base1.yaml",
    "configs/PascalVOC-detection/split3/faster_rcnn_R_101_DoG_FPN_V2/1shot.yaml",
    "configs/PascalVOC-detection/split3/faster_rcnn_R_101_DoG_FPN_V2/3shot.yaml",
    "configs/PascalVOC-detection/split3/faster_rcnn_R_101_DoG_FPN_V2/5shot.yaml",
]

for cfg in configs:
    cmd = [PYTHON_BIN, TRAIN_SCRIPT, "--config-file", cfg]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"❌ Failed on {cfg}\n{proc.stderr}\n")
    else:
        print(f"✅ Completed {cfg}\n")
