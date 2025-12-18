import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 출력 디렉토리 생성
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# 스타일 설정 (논문용 깔끔한 스타일)
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300  # 고해상도
})

# ==========================================
# 1. 탐지 성능 비교 (Detection Performance)
# ==========================================
def plot_detection():
    data = {
        'Method': ['Fusion', 'Baseline', 'No-EO', 'No-Acoustic'],
        'Detection Rate (%)': [96.3, 73.2, 84.7, 89.1],
        'Delay (s)': [4.2, 7.8, 5.9, 5.2],
        'CI_Lower': [3.8, 7.1, 5.4, 4.8],
        'CI_Upper': [4.6, 8.5, 6.4, 5.6]
    }
    df = pd.DataFrame(data)
    
    # 지연 시간 오차 계산 (CI_Upper - Mean)
    yerr = df['CI_Upper'] - df['Delay (s)']

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 막대 그래프 (탐지율)
    color = 'tab:blue'
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Detection Rate (%)', color=color, fontweight='bold')
    bars = sns.barplot(x='Method', y='Detection Rate (%)', data=df, ax=ax1, color='skyblue', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 110)

    # 꺾은선 그래프 (지연 시간)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Detection Delay (s)', color=color, fontweight='bold')
    ax2.errorbar(df['Method'], df['Delay (s)'], yerr=yerr, fmt='-o', color=color, capsize=5, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 10)

    plt.title('Detection Performance: Rate vs. Delay')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure1_detection_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

# ==========================================
# 2. 분류 성능 비교 (F1-Score)
# ==========================================
def plot_classification():
    data = {
        'Method': ['Fusion', 'Baseline', 'No-EO', 'No-Acoustic'],
        'Hostile F1': [0.92, 0.71, 0.81, 0.86],
        'Civilian F1': [0.88, 0.68, 0.75, 0.82]
    }
    df = pd.DataFrame(data)
    df_melt = df.melt(id_vars='Method', var_name='Metric', value_name='Score')

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Method', y='Score', hue='Metric', data=df_melt, palette='viridis')
    plt.ylim(0, 1.0)
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title('Classification Performance (F1 Score)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure2_classification_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

# ==========================================
# 3. 혼동 행렬 (Confusion Matrix)
# ==========================================
def plot_confusion_matrix():
    # Table 3 데이터
    cm = np.array([
        [287, 18, 5],
        [12, 156, 8],
        [8, 14, 48]
    ])
    labels = ['Hostile', 'Civilian', 'Unknown']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Actual Label', fontweight='bold')
    plt.title('Confusion Matrix (Fusion Mode)')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure3_confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

# ==========================================
# 4. 요격 및 유도 성능 비교 (Interception)
# ==========================================
def plot_interception():
    data = {
        'Guidance': ['PN (N=3)', 'Pure Pursuit', 'APN', 'Optimal'],
        'Overall Success (%)': [84.7, 66.9, 87.2, 89.5],
        'Evade Success (%)': [76.3, 45.2, 79.1, 81.4],
        'Miss Distance (m)': [2.1, 5.7, 1.8, 1.5],
        'Miss Dist SD': [0.8, 2.3, 0.7, 0.6]
    }
    df = pd.DataFrame(data)
    
    # 2개의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: 성공률 비교
    df_melt = df.melt(id_vars='Guidance', value_vars=['Overall Success (%)', 'Evade Success (%)'], var_name='Type', value_name='Rate')
    sns.barplot(x='Guidance', y='Rate', hue='Type', data=df_melt, ax=ax1, palette='magma')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Interception Success Rate')

    # 오른쪽: Miss Distance
    sns.barplot(x='Guidance', y='Miss Distance (m)', data=df, ax=ax2, color='salmon', capsize=.1)
    # Error bar 추가
    ax2.errorbar(x=range(len(df)), y=df['Miss Distance (m)'], yerr=df['Miss Dist SD'], fmt='none', c='black', capsize=5)
    ax2.set_ylabel('Avg Miss Distance (m)', fontweight='bold')
    ax2.set_title('Miss Distance Comparison (Lower is Better)')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure4_interception_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

# 실행
if __name__ == "__main__":
    print("Generating Figure 1...")
    plot_detection()
    print("Generating Figure 2...")
    plot_classification()
    print("Generating Figure 3...")
    plot_confusion_matrix()
    print("Generating Figure 4...")
    plot_interception()
    print(f"\nDone! All figures saved to '{output_dir}/' directory.")