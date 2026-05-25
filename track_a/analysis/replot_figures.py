import matplotlib.pyplot as plt
import numpy as np
import os

# Data from Table 4 (User confirmed this is the "reasonable" one)
# Model: [Static ASR, Fuzzing ASR]
data = {
    "GPT-OSS-20B": [3.85, 5.12],
    "Llama-3.2-3B": [17.88, 24.35],
    "DS-Llama-8B": [42.50, 51.60],
    "DS-Qwen-7B": [47.69, 55.10],
    "TAIDE-12B": [60.00, 58.20],
    "Breeze2-8B": [61.92, 59.45],
    "Gemma-4B": [66.35, 68.10],
    "Qwen3-8B": [67.31, 72.40]
}

models = list(data.keys())
static_asr = [data[m][0] for m in models]
fuzzing_asr = [data[m][1] for m in models]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, static_asr, width, label='Track A: Static Narrative', color='#4c72b0')
rects2 = ax.bar(x + width/2, fuzzing_asr, width, label='Track B: Dynamic Fuzzing', color='#c44e52')

# Add axis labels and custom x-axis tick labels.
ax.set_xlabel('Models')
ax.set_ylabel('Attack Success Rate (ASR) %')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout(rect=[0, 0, 1, 0.92])
output_path = r"c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures\comparison_asr.pdf"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Saved figure to {output_path}")
