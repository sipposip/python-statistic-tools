import numpy as np

def bootstrapped_correlation(x, y, perc=5, N=1000):
    assert (len(x) == len(y))
    corrs = []
    for i in range(N):
        indices = np.random.choice(len(x), replace=True, size=len(x))
        corr = np.corrcoef(x[indices], y[indices])[0, 1]
        corrs.append(corr)
    corrs = np.array(corrs)
    meancorr = np.corrcoef(x, y)[0, 1]
    upper = np.percentile(corrs, q=100 - perc)
    lower = np.percentile(corrs, q=perc)

    return meancorr, lower, upper


def bootstrapped_correlation_difference(x1,y1,x2,y2, perc=5, N=1000):
    """ compute the difference between corr(x1,y1) and corr (x2,y2) plus
    bands [perc,100-perc] of the differecne in  correlation, estimated via bootstrpaping.
    This method is useful when the uncertainty between the two correlations is not independent.
    x1,y1,x2,y2 must all have the same length"""
    assert (len(x1) == len(y1))
    assert (len(x1) == len(y2))
    assert (len(x2) == len(x1))
    n_samples = len(x1)
    diff_corrs = []
    # we use the same indices both for sample 1 and sample 2, and then compute the difference in correlation
    for i in range(N):
        indices = np.random.choice(n_samples, replace=True, size=n_samples)
        corr1 = np.corrcoef(x1[indices], y1[indices])[0, 1]
        corr2 = np.corrcoef(x2[indices], y2[indices])[0, 1]
        diff_corrs.append(corr1-corr2)
    corrs = np.array(diff_corrs)
    meancorrdiff = np.corrcoef(x1, y1)[0, 1] - np.corrcoef(x2, y2)[0, 1]
    upper = np.percentile(corrs, q=100 - perc)
    lower = np.percentile(corrs, q=perc)
    assert(upper>=lower)
    return meancorrdiff, lower, upper


def bootstrapped_rmse_difference(x1,x2, perc=5, N=1000):
    """ compute difference between x1 and x2 plus uncertainty, when x1 and x2 are either rmse or standarddeviation

    """
    assert(len(x1)==len(x2))
    n_samples = len(x1)
    means = []
    for i in range(N):
        indices = np.random.choice(n_samples, replace=True, size=n_samples)
        # now compute difference in  RMSE on this subsample
        mm = np.sqrt(np.mean(x1[indices]**2)) - np.sqrt(np.mean(x2[indices]**2))
        means.append(mm)
    means = np.array(means)
    mmean = np.sqrt(np.mean(x1**2)) - np.sqrt(np.mean(x2**2))
    upper = np.percentile(means, q=100 - perc)
    lower = np.percentile(means, q=perc)
    # assert (upper >= lower) # we deactivate this check here because if one or both of x1 and x2
    # concist only of repreated values, then numerical inaccuracis can lead to
    # lower being a tiny little larger than upper (even though they should be the same in this case)
    return np.array([mmean, lower, upper])
