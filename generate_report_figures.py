#!/usr/bin/env python3
"""
보고서용 비교 차트 및 요약 그래프 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def create_scenario_comparison():
    """시나리오별 성능 비교 차트"""
    scenarios = ['Ground', 'Circle', 'Wall', 'Arch']
    coverages = [5.4, 3.6, 2.8, 1.9]
    stds = [0.8, 0.9, None, None]  # Wall, Arch는 single seed
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(scenarios, coverages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Error bars for multi-seed scenarios
    for i, (scenario, coverage, std) in enumerate(zip(scenarios, coverages, stds)):
        if std is not None:
            ax.errorbar(i, coverage, yerr=std, fmt='none', ecolor='black',
                       capsize=5, capthick=2, linewidth=2)

    ax.set_ylabel('최종 커버리지 (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('시나리오', fontsize=14, fontweight='bold')
    ax.set_title('시나리오별 성능 비교 (Multi-Seed)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotate values
    for i, (bar, coverage, std) in enumerate(zip(bars, coverages, stds)):
        height = bar.get_height()
        if std is not None:
            label = f'{coverage:.1f}%\n±{std:.1f}%'
        else:
            label = f'{coverage:.1f}%\n(1 seed)'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/report_scenario_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Scenario comparison saved: figures/report_scenario_comparison.png")
    plt.close()


def create_improvement_comparison():
    """보상 함수 개선 전후 비교"""
    configs = ['개선 전\n(Old Reward)', '개선 후\n(New Reward)']
    coverages = [1.0, 3.2]
    colors = ['#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(configs, coverages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('최종 커버리지 (%)', fontsize=14, fontweight='bold')
    ax.set_title('보상 함수 개선 효과 (Circle 시나리오, 1000 episodes)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotate
    for bar, coverage in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{coverage:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Improvement arrow
    ax.annotate('', xy=(1, 3.0), xytext=(0, 1.2),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(0.5, 2.2, '+220%', fontsize=16, fontweight='bold', color='green',
            ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

    # Config details
    old_text = 'Coverage scale: 100\nProximity scale: 5.0'
    new_text = 'Coverage scale: 500\nProximity scale: 1.0'
    ax.text(0, -0.5, old_text, ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.7))
    ax.text(1, -0.5, new_text, ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.7))

    plt.tight_layout()
    plt.savefig('figures/report_improvement_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Improvement comparison saved: figures/report_improvement_comparison.png")
    plt.close()


def create_multi_seed_analysis():
    """Multi-seed 분석 상세 차트"""
    # Ground scenario data
    ground_seeds = [0, 1, 2, 3, 4]
    ground_coverages = [6.2, 4.9, 5.3, 6.2, 4.4]

    # Circle scenario data
    circle_seeds = [1, 2, 3, 4]
    circle_coverages = [3.0, 3.3, 3.2, 4.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Ground
    ax1.bar(ground_seeds, ground_coverages, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
    mean_ground = np.mean(ground_coverages)
    ax1.axhline(mean_ground, color='red', linestyle='--', linewidth=2, label=f'평균: {mean_ground:.1f}%')
    ax1.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
    ax1.set_ylabel('최종 커버리지 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Ground 시나리오 (300 episodes)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()

    # Annotate
    for seed, cov in zip(ground_seeds, ground_coverages):
        ax1.text(seed, cov + 0.2, f'{cov:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Circle
    ax2.bar(circle_seeds, circle_coverages, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    mean_circle = np.mean(circle_coverages)
    ax2.axhline(mean_circle, color='red', linestyle='--', linewidth=2, label=f'평균: {mean_circle:.1f}%')
    ax2.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
    ax2.set_ylabel('최종 커버리지 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Circle 시나리오 (500 episodes)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    # Annotate
    for seed, cov in zip(circle_seeds, circle_coverages):
        ax2.text(seed, cov + 0.2, f'{cov:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/report_multi_seed_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Multi-seed analysis saved: figures/report_multi_seed_analysis.png")
    plt.close()


def create_statistics_summary_table():
    """통계 요약 테이블 (텍스트 이미지)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    data = [
        ['시나리오', '에피소드', 'Seeds', '평균 커버리지', '표준편차', '최대', '최소', '95% CI'],
        ['Ground', '300', '5 (0-4)', '5.4%', '0.8%', '6.2%', '4.4%', '[3.8%, 7.0%]'],
        ['Circle', '500', '4 (1-4)', '3.6%', '0.9%', '4.9%', '3.0%', '[1.8%, 5.4%]'],
        ['Wall', '500', '1 (0)', '2.8%', '-', '-', '-', '-'],
        ['Arch', '500', '1 (0)', '1.9%', '-', '-', '-', '-'],
        ['', '', '', '', '', '', '', ''],
        ['Baseline (Old)', '1000', '1 (0)', '1.0%', '-', '-', '-', '-'],
    ]

    colors = [['#f0f0f0'] * 8]  # Header
    for i in range(1, len(data)):
        if i == 5:  # Empty row
            colors.append(['white'] * 8)
        elif i == 6:  # Baseline
            colors.append(['#ffe6e6'] * 8)
        elif data[i][0] in ['Ground', 'Circle']:  # Multi-seed
            colors.append(['#e6ffe6'] * 8)
        else:
            colors.append(['white'] * 8)

    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     cellColours=colors, bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Bold header
    for i in range(8):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=12)
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white')

    plt.savefig('figures/report_statistics_table.png', dpi=300, bbox_inches='tight')
    print("✓ Statistics table saved: figures/report_statistics_table.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("보고서용 비교 차트 생성 중...")
    print("="*60 + "\n")

    create_scenario_comparison()
    create_improvement_comparison()
    create_multi_seed_analysis()
    create_statistics_summary_table()

    print("\n" + "="*60)
    print("✅ 모든 보고서 차트 생성 완료!")
    print("="*60)
    print("\n생성된 파일:")
    print("  1. figures/report_scenario_comparison.png")
    print("  2. figures/report_improvement_comparison.png")
    print("  3. figures/report_multi_seed_analysis.png")
    print("  4. figures/report_statistics_table.png")
    print("")
