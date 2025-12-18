import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

# 출력 디렉토리
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# 스타일 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
fig, ax = plt.subplots(figsize=(12, 7))

# 캔버스 설정 (축 숨김)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# ---------------------------------------------------------
# 함수: 박스 그리기
def draw_box(x, y, w, h, text, color='#EDF2F7', edge='#2D3748', title_color='black', subtext=None):
    # 그림자
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1", 
                                    fc='gray', alpha=0.3, ec="none")
    ax.add_patch(shadow)
    # 메인 박스
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                 fc=color, ec=edge, linewidth=2)
    ax.add_patch(box)
    
    # 텍스트
    ax.text(x + w/2, y + h/2 + (0.3 if subtext else 0), text, ha='center', va='center', 
            fontsize=12, fontweight='bold', color=title_color)
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.3, subtext, ha='center', va='center', 
                fontsize=9, color='#4A5568')
    return x, y, w, h

# ---------------------------------------------------------
# 1. Sensors Layer (Left)
# ---------------------------------------------------------
draw_box(0.5, 6.0, 2.0, 1.0, "Radar Sensor", "#FEFCBF", subtext="Range, Azimuth, Vel")
draw_box(0.5, 4.5, 2.0, 1.0, "Acoustic Sensor", "#FEFCBF", subtext="Azimuth (Array)")
draw_box(0.5, 3.0, 2.0, 1.0, "EO/IR Camera", "#FEFCBF", subtext="Class, Image")

ax.text(1.5, 7.5, "Perception Layer\n(Sensors)", ha='center', va='center', fontsize=14, fontweight='bold', color='#744210')

# ---------------------------------------------------------
# 2. Backend Service (Center) - Node.js
# ---------------------------------------------------------
# 큰 컨테이너
bg = patches.FancyBboxPatch((3.5, 2.5), 5.0, 5.0, boxstyle="round,pad=0.2", 
                            fc='white', ec='#2B6CB0', linestyle='--', linewidth=2)
ax.add_patch(bg)
ax.text(6.0, 7.8, "C2 Backend Service (Node.js)", ha='center', va='center', fontsize=14, fontweight='bold', color='#2C5282')

# 내부 모듈
draw_box(4.0, 6.2, 4.0, 0.8, "Sensor Interface & Sim", "#BEE3F8", subtext="Noise Injection / Parsing")
draw_box(4.0, 5.0, 4.0, 0.8, "Multi-Sensor Fusion (EKF)", "#90CDF4", subtext="State Estimation (x, v, a)")
draw_box(4.0, 3.8, 1.9, 0.8, "Threat Eval", "#63B3ED", subtext="Bayesian Score")
draw_box(6.1, 3.8, 1.9, 0.8, "Engagement", "#63B3ED", subtext="PN Guidance")

# ---------------------------------------------------------
# 3. Frontend Layer (Right) - React
# ---------------------------------------------------------
draw_box(9.5, 4.0, 2.0, 2.5, "C2 UI Frontend\n(Tablet)", "#C6F6D5", subtext="React 18 + TS\nReal-time Map\nAlert System")
ax.text(10.5, 7.5, "Presentation Layer\n(Operator)", ha='center', va='center', fontsize=14, fontweight='bold', color='#22543D')

# ---------------------------------------------------------
# 4. Analysis Layer (Bottom) - Python
# ---------------------------------------------------------
draw_box(3.5, 0.5, 5.0, 1.0, "Analysis Pipeline (Python)", "#E9D8FD", subtext="Log Parsing, Performance Metrics, Tuning")

# ---------------------------------------------------------
# 화살표 그리기 (Arrows) - 더 작고 예쁘게
# ---------------------------------------------------------
style = "Simple, tail_width=0.2, head_width=1.8, head_length=3"
kw = dict(arrowstyle=style, color="#4A5568", lw=1.2, alpha=0.8)

# Sensors -> Backend
ax.annotate("", xy=(3.5, 6.6), xytext=(2.6, 6.5), arrowprops=kw)
ax.annotate("", xy=(3.5, 6.6), xytext=(2.6, 5.0), arrowprops=kw)
ax.annotate("", xy=(3.5, 6.6), xytext=(2.6, 3.5), arrowprops=kw)

# Backend Internal
ax.annotate("", xy=(6.0, 5.8), xytext=(6.0, 6.2), arrowprops=kw) # Interface -> Fusion
ax.annotate("", xy=(5.0, 4.6), xytext=(5.0, 5.0), arrowprops=kw) # Fusion -> Threat
ax.annotate("", xy=(7.0, 4.6), xytext=(7.0, 5.0), arrowprops=kw) # Fusion -> Engage
ax.annotate("", xy=(6.1, 4.2), xytext=(5.9, 4.2), arrowprops=kw) # Threat -> Engage

# Backend <-> Frontend (WebSocket) - 양방향 화살표
arrow1 = FancyArrowPatch((8.6, 5.5), (9.4, 5.5), 
                         arrowstyle='->', mutation_scale=12, 
                         color="#E53E3E", lw=1.3, alpha=0.9)
arrow2 = FancyArrowPatch((9.4, 5.5), (8.6, 5.5), 
                         arrowstyle='->', mutation_scale=12, 
                         color="#E53E3E", lw=1.3, alpha=0.9)
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.text(9.0, 5.7, "WebSocket\n(JSON)", ha='center', fontsize=8, color="#E53E3E", fontweight='bold')

# Backend -> Logs -> Analysis
ax.annotate("", xy=(6.0, 1.6), xytext=(6.0, 2.4), 
            arrowprops=dict(arrowstyle="Simple, tail_width=0.2, head_width=1.8, head_length=3", 
                           color="#805AD5", ls=':', lw=1.2, alpha=0.8))
ax.text(6.5, 2.0, "JSONL Logs", ha='center', fontsize=9, color="#805AD5")

# 저장
plt.tight_layout()
output_path = os.path.join(output_dir, 'figure6_system_architecture.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()

print("system architecture 이미지 생성 완료!")