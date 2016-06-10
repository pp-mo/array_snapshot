import numpy as np

dither4x4_0_1 = np.array([
  [1, 9, 3, 11],
  [13, 5, 15, 7],
  [4, 12, 2, 10],
  [16, 8, 14, 6]]) * 1.0 /  17

dither8x8_0_1 = np.array(
    [[ 1, 49, 13, 61,  4, 52, 16, 64],
     [33, 17, 45, 29, 36, 20, 48, 32],
     [ 9, 57,  5, 53, 12, 60,  8, 56],
     [41, 25, 37, 21, 44, 28, 40, 24],
     [ 3, 51, 15, 63,  2, 50, 14, 62],
     [35, 19, 47, 31, 34, 18, 46, 30],
     [11, 59,  7, 55, 10, 58,  6, 54],
     [43, 27, 39, 23, 42, 26, 38, 22]]) * 1.0 / 65

dither_amplitude = 0.2
balanced_dither4x4 = 1.0 + dither_amplitude * 2 * (dither4x4_0_1 - 0.5)
balanced_dither8x8 = 1.0 + dither_amplitude * 2 * (dither8x8_0_1 - 0.5)

def dither_tile(target_shape, pattern=dither8x8_0_1):
    dithers = pattern.flat[:]
    n_dithers = len(dithers)
    ndim = len(target_shape)
    points_per_dim = int(np.floor(np.exp(np.log(n_dithers + 0.1) / ndim)))
    dither_shape = [points_per_dim] * ndim
    n_dithers = np.prod(dither_shape)
    dither = dithers[:n_dithers].reshape(dither_shape)
    return dither

def balanced_dither_tile(shape, amplitude=0.2, pattern=dither8x8_0_1):
    dither = dither_tile(shape, pattern=pattern)
    return balanced(dither, amplitude=amplitude)

def balanced(array, amplitude=0.2):
    d_min, d_max = np.min(array), np.max(array)
    # Normalise to range 0..1
    array = (array - d_min) * 1.0 / (d_max - d_min)
    # Rescale to (1-amp)..(1+amp)
    return 1.0 + amplitude * 2 * (array - 0.5)

def tiled_array(shape, tile=dither8x8_0_1):
    full_muls = [int((shape_dim + tile_dim - 1) / tile_dim)
                 for shape_dim, tile_dim in zip(shape, tile.shape)]
#    print 'full muls:', full_muls
    full_tiles = np.tile(tile, full_muls)
#    print 'full shape:', full_tiles.shape
    part_slices = [slice(0, ndim) for ndim in shape]
#    print 'part slices:', part_slices
    result = full_tiles[part_slices]
    return result

def balanced_dither(shape, amplitude=0.2, pattern=dither8x8_0_1):
    tile = balanced(dither_tile(target_shape=shape, pattern=pattern),
                    amplitude=amplitude)
    result = tiled_array(shape, tile)
    return result

tst = np.array(
    [[11, 12, 13],
     [21, 22, 23]])

tst3 = np.array(
    [[[111, 112, 113, 114],
      [121, 122, 123, 124],
      [131, 132, 133, 134]],
     [[211, 212, 213, 214],
      [221, 222, 223, 224],
      [231, 232, 233, 234]]])

#print tiled_array((7,7), tst)
#print tiled_array((3, 5, 5), tst3)

def show_balanced(array):
    return np.array(1000 * array, dtype=int)

#for ndim in range(1, 7):
#    print
#    print 'N-dims = ', ndim
#    result = show_balanced(dither_tile([1]*ndim))
##    result = np.array(1000 * balanced_dither_tile([1]*ndim), dtype=int)
#    print 'shape = ', result.shape
#    print 'n-vals = ', result.size
#    print result

print show_balanced(balanced_dither((4,3)))
print show_balanced(balanced_dither((3, 5, 7)))
