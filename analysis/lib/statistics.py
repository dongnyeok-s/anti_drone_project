"""
통계 분석 모듈

부트스트랩 신뢰구간, 통계 검정 등 논문용 통계 분석 함수를 제공합니다.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from scipy import stats


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: Callable = np.mean,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    부트스트랩 신뢰구간 계산

    Args:
        data: 데이터 배열
        n_bootstrap: 부트스트랩 샘플 수 (기본: 1000)
        ci: 신뢰수준 (기본: 0.95 = 95%)
        statistic: 통계량 함수 (기본: np.mean)
        random_seed: 재현성을 위한 시드 값

    Returns:
        (하한, 상한) 튜플

    Example:
        >>> data = [0.95, 0.97, 0.94, 0.96, 0.93, 0.98]
        >>> lower, upper = bootstrap_ci(data, ci=0.95)
        >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = np.array(data)
    n = len(data)

    if n == 0:
        return (np.nan, np.nan)

    if n == 1:
        val = statistic(data)
        return (val, val)

    boot_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_samples.append(statistic(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_samples, alpha * 100)
    upper = np.percentile(boot_samples, (1 - alpha) * 100)

    return (float(lower), float(upper))


def paired_ttest(
    baseline: Union[List[float], np.ndarray],
    treatment: Union[List[float], np.ndarray]
) -> Tuple[float, float]:
    """
    Paired t-test 수행

    동일 실험 조건에서 두 방식의 성능 비교에 사용합니다.

    Args:
        baseline: 기준 방식 결과
        treatment: 제안 방식 결과 (동일 순서)

    Returns:
        (t-statistic, p-value) 튜플

    Example:
        >>> baseline = [0.73, 0.75, 0.71, 0.74]
        >>> fusion = [0.96, 0.97, 0.95, 0.96]
        >>> t_stat, p_val = paired_ttest(baseline, fusion)
        >>> print(f"p-value: {p_val:.4f}")
    """
    baseline = np.array(baseline)
    treatment = np.array(treatment)

    if len(baseline) != len(treatment):
        raise ValueError("baseline과 treatment의 길이가 일치해야 합니다")

    if len(baseline) < 2:
        return (np.nan, np.nan)

    t_stat, p_value = stats.ttest_rel(treatment, baseline)
    return (float(t_stat), float(p_value))


def independent_ttest(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    equal_var: bool = False
) -> Tuple[float, float]:
    """
    Independent (Welch's) t-test 수행

    독립된 두 그룹의 평균 비교에 사용합니다.

    Args:
        group1: 첫 번째 그룹 데이터
        group2: 두 번째 그룹 데이터
        equal_var: 등분산 가정 여부 (기본: False = Welch's t-test)

    Returns:
        (t-statistic, p-value) 튜플
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    if len(group1) < 2 or len(group2) < 2:
        return (np.nan, np.nan)

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    return (float(t_stat), float(p_value))


def cohens_d(
    baseline: Union[List[float], np.ndarray],
    treatment: Union[List[float], np.ndarray]
) -> float:
    """
    효과 크기 계산 (Cohen's d)

    두 그룹 간 차이의 실질적 의미를 평가합니다.
    - |d| < 0.2: 작은 효과
    - 0.2 ≤ |d| < 0.5: 중간 효과
    - 0.5 ≤ |d| < 0.8: 큰 효과
    - |d| ≥ 0.8: 매우 큰 효과

    Args:
        baseline: 기준 방식 결과
        treatment: 제안 방식 결과

    Returns:
        Cohen's d 값

    Example:
        >>> baseline = [0.73, 0.75, 0.71, 0.74]
        >>> fusion = [0.96, 0.97, 0.95, 0.96]
        >>> d = cohens_d(baseline, fusion)
        >>> print(f"Cohen's d: {d:.2f} (매우 큰 효과)")
    """
    baseline = np.array(baseline)
    treatment = np.array(treatment)

    n1, n2 = len(baseline), len(treatment)

    if n1 < 2 or n2 < 2:
        return np.nan

    # Pooled standard deviation
    var1, var2 = np.var(baseline, ddof=1), np.var(treatment, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    d = (np.mean(treatment) - np.mean(baseline)) / pooled_std
    return float(d)


def paired_cohens_d(
    baseline: Union[List[float], np.ndarray],
    treatment: Union[List[float], np.ndarray]
) -> float:
    """
    페어드 데이터에 대한 Cohen's d 계산

    Args:
        baseline: 기준 방식 결과
        treatment: 제안 방식 결과 (동일 순서)

    Returns:
        Cohen's d 값
    """
    baseline = np.array(baseline)
    treatment = np.array(treatment)

    if len(baseline) != len(treatment):
        raise ValueError("baseline과 treatment의 길이가 일치해야 합니다")

    diff = treatment - baseline

    if len(diff) < 2:
        return np.nan

    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        return np.nan

    return float(np.mean(diff) / std_diff)


def format_ci(lower: float, upper: float, precision: int = 1) -> str:
    """
    신뢰구간을 논문용 문자열로 포맷팅

    Args:
        lower: 하한
        upper: 상한
        precision: 소수점 자릿수

    Returns:
        "[lower, upper]" 형식의 문자열

    Example:
        >>> format_ci(94.1, 98.2, precision=1)
        '[94.1, 98.2]'
    """
    return f"[{lower:.{precision}f}, {upper:.{precision}f}]"


def format_p_value(p: float, threshold: float = 0.001) -> str:
    """
    p-value를 논문용 문자열로 포맷팅

    Args:
        p: p-value
        threshold: 표시 임계값 (기본: 0.001)

    Returns:
        포맷된 p-value 문자열

    Example:
        >>> format_p_value(0.00023)
        'p < 0.001'
        >>> format_p_value(0.034)
        'p = 0.034'
    """
    if np.isnan(p):
        return "p = N/A"

    if p < threshold:
        return f"p < {threshold}"
    else:
        return f"p = {p:.3f}"


def summarize_with_ci(
    data: Union[List[float], np.ndarray],
    ci: float = 0.95,
    n_bootstrap: int = 1000,
    as_percent: bool = False,
    precision: int = 1
) -> dict:
    """
    데이터의 평균, 표준편차, 신뢰구간을 한번에 계산

    Args:
        data: 데이터 배열
        ci: 신뢰수준
        n_bootstrap: 부트스트랩 샘플 수
        as_percent: 백분율로 표시 여부
        precision: 소수점 자릿수

    Returns:
        {
            'mean': 평균,
            'std': 표준편차,
            'ci_lower': 하한,
            'ci_upper': 상한,
            'ci_str': "[lower, upper]" 문자열,
            'n': 샘플 수
        }
    """
    data = np.array(data)
    n = len(data)

    if n == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_str': '[N/A]',
            'n': 0
        }

    multiplier = 100 if as_percent else 1

    mean = float(np.mean(data)) * multiplier
    std = float(np.std(data, ddof=1)) * multiplier if n > 1 else 0.0

    lower, upper = bootstrap_ci(data, n_bootstrap=n_bootstrap, ci=ci)
    lower *= multiplier
    upper *= multiplier

    return {
        'mean': round(mean, precision),
        'std': round(std, precision),
        'ci_lower': round(lower, precision),
        'ci_upper': round(upper, precision),
        'ci_str': format_ci(lower, upper, precision),
        'n': n
    }


def compare_methods(
    baseline: Union[List[float], np.ndarray],
    treatment: Union[List[float], np.ndarray],
    paired: bool = True,
    ci: float = 0.95,
    as_percent: bool = False
) -> dict:
    """
    두 방식 비교 통계 요약

    Args:
        baseline: 기준 방식 결과
        treatment: 제안 방식 결과
        paired: 페어드 데이터 여부
        ci: 신뢰수준
        as_percent: 백분율로 표시 여부

    Returns:
        통계 요약 딕셔너리
    """
    baseline_summary = summarize_with_ci(baseline, ci=ci, as_percent=as_percent)
    treatment_summary = summarize_with_ci(treatment, ci=ci, as_percent=as_percent)

    if paired:
        t_stat, p_value = paired_ttest(baseline, treatment)
        d = paired_cohens_d(baseline, treatment)
    else:
        t_stat, p_value = independent_ttest(baseline, treatment)
        d = cohens_d(baseline, treatment)

    return {
        'baseline': baseline_summary,
        'treatment': treatment_summary,
        'improvement': treatment_summary['mean'] - baseline_summary['mean'],
        't_statistic': round(t_stat, 3) if not np.isnan(t_stat) else None,
        'p_value': p_value,
        'p_value_str': format_p_value(p_value),
        'cohens_d': round(d, 2) if not np.isnan(d) else None,
        'effect_size': _interpret_cohens_d(d),
        'significant': p_value < 0.05 if not np.isnan(p_value) else None
    }


def _interpret_cohens_d(d: float) -> str:
    """Cohen's d 해석"""
    if np.isnan(d):
        return "N/A"

    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# 논문용 테이블 생성 헬퍼
def generate_latex_row(
    label: str,
    mean: float,
    ci_str: str,
    is_best: bool = False,
    include_ci: bool = True
) -> str:
    """
    LaTeX 테이블 행 생성

    Args:
        label: 행 레이블
        mean: 평균값
        ci_str: 신뢰구간 문자열
        is_best: 최고 성능 여부 (볼드 처리)
        include_ci: CI 포함 여부

    Returns:
        LaTeX 테이블 행 문자열
    """
    if is_best:
        mean_str = f"\\textbf{{{mean}}}"
        label_str = f"\\textbf{{{label}}}"
    else:
        mean_str = str(mean)
        label_str = label

    if include_ci:
        return f"{label_str} & {mean_str} & {ci_str} \\\\"
    else:
        return f"{label_str} & {mean_str} \\\\"
