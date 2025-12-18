import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 출력 디렉토리 생성
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'sans-serif', 'figure.dpi': 300})

def simulate_engagement():
    dt = 0.05
    time = np.arange(0, 10, dt)
    
    # 적 드론 (Target) - 회피 기동 (Sine Wave)
    t_x = 500 - 30 * time  # X축으로 접근
    t_y = 200 + 50 * np.sin(time) # Y축으로 흔들며 접근 (EVADE Mode)
    
    # 1. Pure Pursuit (PP) 시뮬레이션
    pp_x, pp_y = [0], [0] # 요격 드론 시작 위치 (0,0)
    vp = 40 # 요격 드론 속도 (m/s)
    
    for i in range(len(time)-1):
        dx = t_x[i] - pp_x[-1]
        dy = t_y[i] - pp_y[-1]
        dist = np.hypot(dx, dy)
        
        if dist < 5: break # 요격 성공
        
        # PP는 항상 표적을 향해 직진
        vx = vp * (dx / dist)
        vy = vp * (dy / dist)
        
        pp_x.append(pp_x[-1] + vx * dt)
        pp_y.append(pp_y[-1] + vy * dt)
        
    # 2. Proportional Navigation (PN) 시뮬레이션
    pn_x, pn_y = [0], [0]
    pn_vx, pn_vy = vp * np.cos(0.5), vp * np.sin(0.5) # 초기 속도 벡터
    N = 3 # 유도 상수
    
    for i in range(len(time)-1):
        dx = t_x[i] - pn_x[-1]
        dy = t_y[i] - pn_y[-1]
        dist = np.hypot(dx, dy)
        
        if dist < 2: break # 요격 성공
        
        # LOS 각도 및 각속도 계산
        los_angle = np.arctan2(dy, dx)
        if i > 0:
            prev_dx = t_x[i-1] - pn_x[-2]
            prev_dy = t_y[i-1] - pn_y[-2]
            prev_los = np.arctan2(prev_dy, prev_dx)
            los_rate = (los_angle - prev_los) / dt
        else:
            los_rate = 0
            
        # PN 가속도 명령 (a = N * V_closing * LOS_rate)
        # 간단한 모델링을 위해 속도 벡터 회전으로 구현
        accel_cmd = N * vp * los_rate
        
        vel_angle = np.arctan2(pn_vy, pn_vx)
        vel_angle += accel_cmd * dt / vp # 속도 벡터 회전
        
        pn_vx = vp * np.cos(vel_angle)
        pn_vy = vp * np.sin(vel_angle)
        
        pn_x.append(pn_x[-1] + pn_vx * dt)
        pn_y.append(pn_y[-1] + pn_vy * dt)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # 적 드론 궤적
    plt.plot(t_x[:len(pp_x)], t_y[:len(pp_x)], 'k--', label='Hostile Drone (Evading)', linewidth=2, alpha=0.6)
    
    # 유도탄 궤적
    plt.plot(pp_x, pp_y, 'r-.', label='Pure Pursuit (Miss)', linewidth=2)
    plt.plot(pn_x, pn_y, 'b-', label='PN Guidance (Hit)', linewidth=2.5)
    
    # 충돌 지점 표시
    plt.scatter(pn_x[-1], pn_y[-1], color='blue', s=100, marker='*', label='PN Intercept', zorder=5)
    plt.scatter(pp_x[-1], pp_y[-1], color='red', s=50, marker='x', label='PP Miss', zorder=5)

    plt.title('Engagement Trajectory: PN vs Pure Pursuit against Evading Target', fontsize=14, fontweight='bold')
    plt.xlabel('Downrange (m)', fontsize=12)
    plt.ylabel('Crossrange (m)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure5_engagement_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating engagement trajectory plot...")
    simulate_engagement()
    print(f"Done! Figure saved to '{output_dir}/' directory.")