#!/usr/bin/env python3
"""
실험 결과 분석 스크립트
모든 실험 결과에서 최종 커버리지를 추출하고 통계 분석
"""

import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def extract_coverage_from_image(image_path):
    """학습 곡선 이미지에서 최종 커버리지 값을 텍스트로부터 추출"""
    # 이미지를 읽어서 텍스트 영역을 확인
    # 실제로는 저장된 history 데이터가 없으므로, 디렉토리 이름과 수동 확인 필요
    return None

def parse_results_manual():
    """수동으로 확인한 결과를 정리"""
    results = {
        'ground': {
            'seed0': {'dir': 'run_20251207_133414_ground_seed0_ep300', 'coverage': 0.062},
            'seed1': {'dir': 'run_20251207_135529_ground_seed1_ep300', 'coverage': None},
            'seed2': {'dir': 'run_20251207_135701_ground_seed2_ep300', 'coverage': None},
            'seed3': {'dir': 'run_20251207_135820_ground_seed3_ep300', 'coverage': None},
            'seed4': {'dir': 'run_20251207_135941_ground_seed4_ep300', 'coverage': None},
        },
        'circle': {
            'seed1': {'dir': 'run_20251207_141549_circle_seed1_ep500', 'coverage': None},
            'seed2': {'dir': 'run_20251207_141802_circle_seed2_ep500', 'coverage': None},
            'seed3': {'dir': 'run_20251207_142013_circle_seed3_ep500', 'coverage': None},
            'seed4': {'dir': 'run_20251207_142642_circle_seed4_ep500', 'coverage': None},
        },
        'baseline': {
            'old_reward': {'dir': 'run_20251207_125040_seed0_ep1000', 'coverage': 0.010},
        }
    }
    return results

def list_all_runs():
    """figures 디렉토리의 모든 실행 결과 나열"""
    figures_dir = Path('figures')
    runs = sorted([d for d in figures_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])

    print("모든 실험 실행 결과:")
    print("=" * 80)
    for run_dir in runs:
        learning_curve = run_dir / 'learning_curves.png'
        animation = run_dir / 'particle_animation.gif'
        has_curve = '✓' if learning_curve.exists() else '✗'
        has_anim = '✓' if animation.exists() else '✗'
        print(f"{run_dir.name:60s} [curve:{has_curve}] [anim:{has_anim}]")

    return runs

if __name__ == '__main__':
    print("실험 결과 분석 시작\n")
    runs = list_all_runs()
    print(f"\n총 {len(runs)}개의 실험 실행 결과 발견")
