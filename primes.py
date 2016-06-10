import numpy as np


_primes_upto_somewhere = np.array([2])

def _calculate_primes_upto(n):
    global _primes_upto_somewhere
    slots = np.array([0, 0] + [1] * n)
    _primes_upto_somewhere = list(_primes_upto_somewhere)
    for prime in _primes_upto_somewhere:
        slots[0::prime] = 0
    while np.any(slots):
        prime = np.nonzero(slots[1:])[0][0] + 1
        slots[0::prime] = 0
        _primes_upto_somewhere += [prime]
    _primes_upto_somewhere = np.array(_primes_upto_somewhere)

def largest_prime_below(n):
    if n > _primes_upto_somewhere[-1]:
        _calculate_primes_upto(n)
    i_at = np.nonzero(_primes_upto_somewhere <= n)[0][-1]
    result = _primes_upto_somewhere[i_at]
    return result

for i_n in range(1, 3):
    n = 10 ** i_n
    p = largest_prime_below(n)
    print
    print 'upto {} is {}'.format(n, p)
    print 'new list : ', _primes_upto_somewhere


p = [2, 3]
for _ in range(10):
  print p
  p += [np.prod(p)-1]
