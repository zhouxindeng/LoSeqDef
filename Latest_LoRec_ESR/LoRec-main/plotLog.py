import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_loss(log_path):
    # 数据存储
    epochs, rec_losses, lct_losses = [], [], []

    # 读取日志
    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(r'Epoch (\d+): Rec Loss: ([\d\.-]+), LCT Loss: ([\d\.-]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                rec_losses.append(float(match.group(2)))
                lct_losses.append(float(match.group(3)))

    # 创建画布
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=120)
    
    # 绘制Rec Loss（左轴）
    ax1.set_xlabel('Epoch', fontsize=12, labelpad=10)
    ax1.set_ylabel('Recommendation Loss', color='#1f77b4', fontsize=12)
    rec_line = ax1.plot(epochs, rec_losses, color='#1f77b4', linewidth=1.5, label='Rec Loss')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0, max(rec_losses)*1.1)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 创建右轴绘制LCT Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('LCT Loss', color='#d62728', fontsize=12)  
    lct_line = ax2.plot(epochs, lct_losses, color='#d62728', linewidth=1.5, label='LCT Loss')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(min(lct_losses)*1.1, max(lct_losses)*0.9)

    # 合并图例
    lines = rec_line + lct_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', 
              bbox_to_anchor=(0.85, 1.15), fontsize=10, frameon=True)

    # 美化显示
    plt.title('Training Loss Analysis\n on Arts Dataset', 
             fontsize=14, pad=25, fontweight='semibold')
    plt.xticks(range(0, 101, 10))
    fig.tight_layout()

    # 保存输出
    output_path = str(Path(log_path).parent / "dual_axis_loss.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# 使用示例
log_path = "F:/GraduationProject/GraduationProject/Latest_LoRec/LoRec-main/log/SASrec/Llama2_13/Arts/LCT/20250321/log_attack_random_1_202503212315.txt"
plot_training_loss(log_path)