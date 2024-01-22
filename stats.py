import scipy


def mannwhitney(group1, group2, metric):

    n1 = len(group1)
    n2 = len(group2)

    df = (n1 * n2) / (n1 + n2 - 1)

    stats_res = scipy.stats.mannwhitneyu(group1, group2)
    if stats_res.pvalue < 0.001:
        pvalue = "$<$0.001"
    else:
        pvalue = f"={stats_res.pvalue:.2f}"

    print(f"{metric}: U({df:.0f})={stats_res.statistic:.1f}, p{pvalue}")


def wilcoxon(group1, group2, metric, return_string=False):

    n1 = len(group1)
    n2 = len(group2)

    assert n1 == n2

    stats_res = scipy.stats.wilcoxon(group1, group2)
    if stats_res.pvalue < 0.001:
        pvalue = "$<$0.001"
    else:
        pvalue = f"={stats_res.pvalue:.2f}"

    if return_string:
        return f"{metric}: W({n1:.0f})={stats_res.statistic:.1f}, p{pvalue}"
    else:
        print(f"{metric}: W({n1:.0f})={stats_res.statistic:.1f}, p{pvalue}")


def spearmanr(group1, group2, metric):

    n1 = len(group1)
    n2 = len(group2)

    assert n1 == n2

    stats_res = scipy.stats.spearmanr(group1, group2)

    if stats_res.pvalue < 0.001:
        pvalue = "$<$0.001"
    else:
        pvalue = f"={stats_res.pvalue:.2f}"

    print(f"{metric}: $\\rho$({n1:.0f})={stats_res.statistic:.1f}, p{pvalue}")
