import numpy as np


dims = (10, 71, 744, )

test_dims = [
    (2, 2, 2, 2),
    (1, 2, 12, 16),
    (10, 100, 1000, 10000),
    (2, 71, 744, 928),
    (1, 2, 1152, 1536),
    (10, 70, 1153, 1536),
]


def lower_upper(shape):
    lower_points = [0 if ndim<4 else int(np.floor((ndim - 1) * 1.0 / 6) + 0.5)
                    for ndim in shape]
    upper_points = [ndim-1 if ndim<4 else int(np.ceil((ndim - 1) * 5.0 / 6) + 0.5)
                    for ndim in shape]
    return lower_points, upper_points


def _corner_inds(shape):
    if len(shape) == 0:
        yield ()
    else:
        ndim = shape[-1]
        lower_ind = 0 if ndim<4 else int(np.floor((ndim - 1) * 1.0 / 6) + 0.5)
        upper_ind = ndim-1 if ndim<4 else int(np.ceil((ndim - 1) * 5.0 / 6) + 0.5)
        for rest_shape in _corner_inds(shape[:-1]):
            yield rest_shape + (lower_ind,)
            if upper_ind != lower_ind:
                yield rest_shape + (upper_ind,)


def _edge_inds(shape):
    halfway_points = [ 0 if ndim<2 else int(ndim/2) - 1
                      for ndim in shape]
    for i_dim, ndim in enumerate(shape):
        inds = halfway_points[:]
        inds[i_dim] = 0
        yield tuple(inds[:])
        if ndim > 1:
            inds[i_dim] = ndim - 1
            yield tuple(inds[:])


from magic_divisors import magic_divisor_closeto

def _sparse_inds(shape, n_samples=16):
    n_size = np.prod(shape)
    n_ideal = max(2, n_size * 1.0 / n_samples)
    n_step = magic_divisor_closeto(int(n_ideal))
    indices = np.arange(n_step/2, n_size, n_step)
    results = np.unravel_index(indices, shape)
    results = np.array(results).T
    return [tuple(inds) for inds in list(results)]


def sample_indices(shape):
    return (list(_corner_inds(shape)) +
            list(_edge_inds(shape)) + 
            list(_sparse_inds(shape)))


#shape = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
#print 'test shape : ', shape
#lower_inds, upper_inds = lower_upper(shape)
#print '  lower inds: ', lower_inds
#print '  upper inds: ', upper_inds
#
#print
#print 'TEST EDGE INDS'
#print '=============='
#
#for shape in test_dims:
#    print
#    print 'test shape : ', shape
#    print 'edge inds:'
#    result = list(_edge_inds(shape))
#    for inds in result:
#        print '  ', inds
#
#print
#print 'TEST CORNER INDS'
#print '================'
#
#for shape in test_dims:
#    print
#    print 'test shape : ', shape
#    print 'corner inds:'
#    result = list(_corner_inds(shape))
#    for inds in result:
#        print '  ', inds

#print
#print 'TEST SPARSE INDS (16)'
#print '====================='
#
#for shape in test_dims:
#    print
#    print 'test shape : ', shape
#    print 'sparse inds:'
#    result = _sparse_inds(shape)
#    for inds in result:
#        print '  ', inds

print
print 'TEST SAMPLE INDS (16)'
print '====================='

for shape in test_dims:
    print
    print 'test shape : ', shape
    print 'sample indices:'
    result = sample_indices(shape)
    for inds in result:
        print '  ', inds
        inds = np.array(inds)
        assert np.all(inds >= 0)
        assert np.all(inds < shape)
