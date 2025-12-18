"""
ROC/PR Curve 분석 모듈

Youden Index 기반 최적 임계값 계산 및 ROC/PR 분석 함수를 제공합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ROCPoint:
    """ROC Curve 상의 한 점"""
    threshold: float
    tpr: float  # True Positive Rate (Sensitivity, Recall)
    fpr: float  # False Positive Rate (1 - Specificity)
    tnr: float  # True Negative Rate (Specificity)
    fnr: float  # False Negative Rate (Miss Rate)

    @property
    def youden_index(self) -> float:
        """Youden Index (J = TPR - FPR = Sensitivity + Specificity - 1)"""
        return self.tpr - self.fpr


@dataclass
class PRPoint:
    """PR Curve 상의 한 점"""
    threshold: float
    precision: float
    recall: float  # = TPR

    @property
    def f1_score(self) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class OptimalThreshold:
    """최적 임계값 분석 결과"""
    threshold: float
    youden_index: float
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1_score: float


def compute_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    thresholds: Optional[List[float]] = None
) -> List[ROCPoint]:
    """
    ROC Curve 계산

    Args:
        y_true: 실제 레이블 (0 또는 1)
        y_scores: 예측 점수 (0~100 또는 0~1)
        thresholds: 평가할 임계값 목록 (기본: 0~100, 1단위)

    Returns:
        ROCPoint 리스트

    Example:
        >>> y_true = [1, 1, 0, 1, 0, 0, 1, 0]
        >>> y_scores = [90, 85, 30, 75, 40, 20, 80, 35]
        >>> roc = compute_roc_curve(y_true, y_scores)
        >>> best = max(roc, key=lambda p: p.youden_index)
        >>> print(f"최적 임계값: {best.threshold}, J={best.youden_index:.3f}")
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if thresholds is None:
        # 점수가 0~1 범위인지 0~100 범위인지 판단
        if y_scores.max() <= 1.0:
            thresholds = np.linspace(0, 1, 101)
        else:
            thresholds = np.arange(0, 101, 1)

    total_positive = np.sum(y_true == 1)
    total_negative = np.sum(y_true == 0)

    roc_points = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0
        tnr = tn / total_negative if total_negative > 0 else 0
        fnr = fn / total_positive if total_positive > 0 else 0

        roc_points.append(ROCPoint(
            threshold=float(thresh),
            tpr=float(tpr),
            fpr=float(fpr),
            tnr=float(tnr),
            fnr=float(fnr)
        ))

    return roc_points


def compute_pr_curve(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    thresholds: Optional[List[float]] = None
) -> List[PRPoint]:
    """
    Precision-Recall Curve 계산

    Args:
        y_true: 실제 레이블 (0 또는 1)
        y_scores: 예측 점수
        thresholds: 평가할 임계값 목록

    Returns:
        PRPoint 리스트
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if thresholds is None:
        if y_scores.max() <= 1.0:
            thresholds = np.linspace(0, 1, 101)
        else:
            thresholds = np.arange(0, 101, 1)

    total_positive = np.sum(y_true == 1)
    pr_points = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_positive if total_positive > 0 else 0

        pr_points.append(PRPoint(
            threshold=float(thresh),
            precision=float(precision),
            recall=float(recall)
        ))

    return pr_points


def compute_youden_optimal_threshold(
    roc_points: List[ROCPoint]
) -> Tuple[float, float]:
    """
    Youden Index 기반 최적 임계값 계산

    Youden Index J = TPR - FPR = Sensitivity + Specificity - 1
    J를 최대화하는 임계값이 최적 임계값입니다.

    Args:
        roc_points: ROCPoint 리스트

    Returns:
        (최적 임계값, 최대 Youden Index) 튜플

    Example:
        >>> roc = compute_roc_curve(y_true, y_scores)
        >>> threshold, j = compute_youden_optimal_threshold(roc)
        >>> print(f"최적 임계값: {threshold}, J = {j:.3f}")
    """
    if not roc_points:
        return (0, 0)

    best_point = max(roc_points, key=lambda p: p.youden_index)
    return (best_point.threshold, best_point.youden_index)


def compute_f1_optimal_threshold(
    pr_points: List[PRPoint]
) -> Tuple[float, float]:
    """
    F1 Score 기반 최적 임계값 계산

    Args:
        pr_points: PRPoint 리스트

    Returns:
        (최적 임계값, 최대 F1 Score) 튜플
    """
    if not pr_points:
        return (0, 0)

    best_point = max(pr_points, key=lambda p: p.f1_score)
    return (best_point.threshold, best_point.f1_score)


def find_optimal_threshold(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    method: str = 'youden'
) -> OptimalThreshold:
    """
    최적 임계값 종합 분석

    Args:
        y_true: 실제 레이블
        y_scores: 예측 점수
        method: 최적화 방법 ('youden' 또는 'f1')

    Returns:
        OptimalThreshold 객체

    Example:
        >>> y_true = [1, 1, 0, 1, 0, 0, 1, 0]
        >>> y_scores = [90, 85, 30, 75, 40, 20, 80, 35]
        >>> opt = find_optimal_threshold(y_true, y_scores)
        >>> print(f"최적 임계값: {opt.threshold}")
        >>> print(f"Youden Index: {opt.youden_index:.3f}")
        >>> print(f"F1 Score: {opt.f1_score:.3f}")
    """
    roc_points = compute_roc_curve(y_true, y_scores)
    pr_points = compute_pr_curve(y_true, y_scores)

    if method == 'youden':
        optimal_thresh, _ = compute_youden_optimal_threshold(roc_points)
    else:
        optimal_thresh, _ = compute_f1_optimal_threshold(pr_points)

    # 해당 임계값에서의 모든 지표 계산
    roc_point = next((p for p in roc_points if p.threshold == optimal_thresh), roc_points[0])
    pr_point = next((p for p in pr_points if p.threshold == optimal_thresh), pr_points[0])

    return OptimalThreshold(
        threshold=optimal_thresh,
        youden_index=roc_point.youden_index,
        tpr=roc_point.tpr,
        fpr=roc_point.fpr,
        precision=pr_point.precision,
        recall=pr_point.recall,
        f1_score=pr_point.f1_score
    )


def compute_auc(roc_points: List[ROCPoint]) -> float:
    """
    AUC (Area Under ROC Curve) 계산

    트라페조이드 규칙을 사용한 수치 적분

    Args:
        roc_points: ROCPoint 리스트

    Returns:
        AUC 값 (0~1)
    """
    if not roc_points:
        return 0.0

    # FPR 기준으로 정렬
    sorted_points = sorted(roc_points, key=lambda p: p.fpr)

    auc = 0.0
    for i in range(1, len(sorted_points)):
        x_diff = sorted_points[i].fpr - sorted_points[i - 1].fpr
        y_avg = (sorted_points[i].tpr + sorted_points[i - 1].tpr) / 2
        auc += x_diff * y_avg

    return float(auc)


def compute_auc_pr(pr_points: List[PRPoint]) -> float:
    """
    AUC-PR (Area Under PR Curve) 계산

    Args:
        pr_points: PRPoint 리스트

    Returns:
        AUC-PR 값 (0~1)
    """
    if not pr_points:
        return 0.0

    # Recall 기준으로 정렬
    sorted_points = sorted(pr_points, key=lambda p: p.recall)

    auc = 0.0
    for i in range(1, len(sorted_points)):
        x_diff = sorted_points[i].recall - sorted_points[i - 1].recall
        y_avg = (sorted_points[i].precision + sorted_points[i - 1].precision) / 2
        auc += x_diff * y_avg

    return float(auc)


def analyze_threshold_sensitivity(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    threshold_range: Tuple[float, float] = (60, 90),
    step: float = 5
) -> List[Dict]:
    """
    임계값 민감도 분석

    다양한 임계값에서의 성능 변화를 분석합니다.

    Args:
        y_true: 실제 레이블
        y_scores: 예측 점수
        threshold_range: 분석할 임계값 범위
        step: 임계값 간격

    Returns:
        각 임계값별 성능 지표 딕셔너리 리스트
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    results = []

    total_positive = np.sum(y_true == 1)
    total_negative = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

        results.append({
            'threshold': float(thresh),
            'tpr': round(float(tpr), 4),
            'fpr': round(float(fpr), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(tpr), 4),
            'f1_score': round(float(f1), 4),
            'youden_index': round(float(tpr - fpr), 4),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        })

    return results


def roc_to_dict(roc_points: List[ROCPoint]) -> Dict:
    """ROC 데이터를 딕셔너리로 변환 (JSON 저장용)"""
    return {
        'thresholds': [p.threshold for p in roc_points],
        'tpr': [p.tpr for p in roc_points],
        'fpr': [p.fpr for p in roc_points],
        'youden_index': [p.youden_index for p in roc_points]
    }


def pr_to_dict(pr_points: List[PRPoint]) -> Dict:
    """PR 데이터를 딕셔너리로 변환 (JSON 저장용)"""
    return {
        'thresholds': [p.threshold for p in pr_points],
        'precision': [p.precision for p in pr_points],
        'recall': [p.recall for p in pr_points],
        'f1_score': [p.f1_score for p in pr_points]
    }


# 논문용 출력 헬퍼
def format_threshold_analysis(optimal: OptimalThreshold) -> str:
    """최적 임계값 분석 결과를 논문용 텍스트로 포맷팅"""
    return f"""최적 임계값 분석 결과:
- 임계값: {optimal.threshold}점
- Youden Index (J): {optimal.youden_index:.3f}
- TPR (Sensitivity): {optimal.tpr:.1%}
- FPR (1-Specificity): {optimal.fpr:.1%}
- Precision: {optimal.precision:.1%}
- Recall: {optimal.recall:.1%}
- F1 Score: {optimal.f1_score:.3f}

Youden Index J = TPR - FPR = {optimal.tpr:.3f} - {optimal.fpr:.3f} = {optimal.youden_index:.3f}
"""
