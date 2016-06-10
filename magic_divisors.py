import numpy as np


magic_prime = 11
_early_divisors = np.array([2, 3, 6, 9, 13, 17])
_chosen_divisors = np.array([2, 3, 11, 17, 31, 59, 97])
  # N.B. we don't like either 5 or 7 very much as 70 is a common number of levels


#magic_primes = [2, 3, 11]
#def _expand_divisors():
#    global _chosen_divisors
#    _chosen_divisors = list(_chosen_divisors)
#    for prime in magic_primes:
#        existing_max = _chosen_divisors[-1]
#        new_divisors = [x * prime for x in _chosen_divisors]
#        new_divisors = [x for x in new_divisors
#                        if x > existing_max and x not in _chosen_divisors]
#        _chosen_divisors += new_divisors
#    _chosen_divisors = np.array(_chosen_divisors)


def _expand_divisors():
    global _chosen_divisors
    _chosen_divisors = list(_chosen_divisors)
    existing_max = _chosen_divisors[-1]
    new_divisors = [x * magic_prime for x in _chosen_divisors]
    new_divisors = [x for x in new_divisors
                    if x > existing_max and x not in _chosen_divisors]
    _chosen_divisors += new_divisors
    _chosen_divisors = np.array(_chosen_divisors)


def magic_divisor_closeto(n):
    if n <= _early_divisors[-1]:
        choosefrom = _early_divisors
    else:
        while n > _chosen_divisors[-1]:
            _expand_divisors()
        choosefrom = _chosen_divisors
    i_at = np.nonzero(choosefrom <= n)[0][-1]
    result = choosefrom[i_at]
    return result

if __name__ == '__main__':
    for n in [4, 8, 16, 30, 50, 75] + [10**i_n for i_n in range(2, 12)]:
        p = magic_divisor_closeto(n)
        print
        print 'upto {} (~10^{}) is {}'.format(n, int(np.log10(n)), p)
        current_set = list(_chosen_divisors)
        current_set = [x for x in _early_divisors
                       if x not in current_set] + current_set
        current_set = sorted(current_set)
        print 'new list : ', np.array(current_set)

    current_set = np.array(current_set, dtype=float)
    ratios = current_set[1:] / current_set[:-1]
    print
    print 'ratios:'
    print ratios

    assert np.all(ratios <= 2.0)
