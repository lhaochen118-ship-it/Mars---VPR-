/* Thin C shim over VLFeat 0.9.21 (libvl.so) reproducing, bit for bit, what the MATLAB MEX files
 * vl_imsmooth.c and vl_dsift.c compute when called by vl_phow.m with its default options.
 * MATLAB stores an M x N image column-major; the MEX files hand that buffer to VLFeat as an image of
 * width M and height N (i.e. transposed) and transpose the descriptors back. We do exactly the same, so the
 * caller passes the image as a column-major (Fortran-order) float32 buffer of M rows and N columns.
 * Build: gcc -O2 -shared -fPIC -I<vlfeat> phow_shim.c -L<dir of libvl.so> -lvl -Wl,-rpath,'$ORIGIN' -o libphow.so
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vl/dsift.h>
#include <vl/imopv.h>
#include <vl/mathop.h>

/* MATLAB: J = vl_imsmooth(I, sigma)  (Gaussian kernel, padding by continuity, subsample 1) */
int mex_imsmooth(float *out, const float *in, int M, int N, double sigma)
{
  vl_size W ;
  vl_index j ;
  float *filter, *tmp ;
  float acc = 0 ;
  int flags = VL_PAD_BY_CONTINUITY | VL_TRANSPOSE ;
  if (sigma < 0.01) { memcpy(out, in, sizeof(float) * (size_t)M * N) ; return 0 ; }
  W = (vl_size) ceil(4.0 * sigma) ;
  filter = (float*) malloc(sizeof(float) * (2 * W + 1)) ;
  tmp = (float*) malloc(sizeof(float) * (size_t)M * N) ;
  if (!filter || !tmp) return -1 ;
  for (j = 0 ; j < (signed)(2 * W + 1) ; ++j) {
    float z = ((float) j - W) / (sigma + VL_EPSILON_F) ;
    filter[j] = exp(- 0.5 * (z * z)) ;
    acc += filter[j] ;
  }
  for (j = 0 ; j < (signed)(2 * W + 1) ; ++j) filter[j] /= acc ;
  /* numRows = M, numColumns = N, numOutputRows = M, numOutputColumns = N (step 1) */
  vl_imconvcol_vf(tmp, N, in, M, N, M, filter, -(vl_index)W, (vl_index)W, 1, flags) ;
  vl_imconvcol_vf(out, M, tmp, N, M, N, filter, -(vl_index)W, (vl_index)W, 1, flags) ;
  free(filter) ; free(tmp) ;
  return 0 ;
}

/* MATLAB: [f, d] = vl_dsift(I, 'norm', 'windowsize', ws, ['fast'], 'step', step, 'size', binSize,
 *                           'bounds', [off off +inf +inf])   (uint8 descriptors, as vl_phow's default)
 * Returns the number of frames n (or -n if maxFrames is too small). frames: n x 3 (x, y, norm), 1-based MATLAB
 * coordinates; descr: n x 128 uint8 in MATLAB's (transposed back) layout. */
int mex_dsift(const float *data, int M, int N, int binSize, int step, double off, double windowSize, int fast,
              int maxFrames, double *frames, unsigned char *descr)
{
  VlDsiftDescriptorGeometry geom ;
  VlDsiftFilter *dsift ;
  VlDsiftKeypoint const *kp ;
  float const *ds ;
  float tmp[128] ;
  double b[4] ;
  int n, k, i, descrSize ;
  geom.numBinT = 8 ; geom.numBinX = 4 ; geom.numBinY = 4 ; geom.binSizeX = binSize ; geom.binSizeY = binSize ;
  dsift = vl_dsift_new(M, N) ;
  vl_dsift_set_geometry(dsift, &geom) ;
  vl_dsift_set_steps(dsift, step, step) ;
  b[0] = off - 1 ; b[1] = off - 1 ; b[2] = INFINITY ; b[3] = INFINITY ;   /* MEX: bounds - 1 (inf - 1 = inf) */
  vl_dsift_set_bounds(dsift, (int) VL_MAX(b[1], 0), (int) VL_MAX(b[0], 0),
                      (int) VL_MIN(b[3], M - 1), (int) VL_MIN(b[2], N - 1)) ;
  vl_dsift_set_flat_window(dsift, fast) ;
  if (windowSize >= 0) vl_dsift_set_window_size(dsift, windowSize) ;
  n = vl_dsift_get_keypoint_num(dsift) ;
  descrSize = vl_dsift_get_descriptor_size(dsift) ;
  if (n > maxFrames || descrSize != 128) { vl_dsift_delete(dsift) ; return -n ; }
  vl_dsift_process(dsift, data) ;
  kp = vl_dsift_get_keypoints(dsift) ;
  ds = vl_dsift_get_descriptors(dsift) ;
  geom = *vl_dsift_get_geometry(dsift) ;
  for (k = 0 ; k < n ; ++k) {
    frames[3 * k + 0] = kp[k].y + 1 ;
    frames[3 * k + 1] = kp[k].x + 1 ;
    frames[3 * k + 2] = kp[k].norm ;
    vl_dsift_transpose_descriptor(tmp, ds + descrSize * k, geom.numBinT, geom.numBinX, geom.numBinY) ;
    for (i = 0 ; i < descrSize ; ++i) descr[(size_t)k * 128 + i] = (vl_uint8) (VL_MIN(512.0F * tmp[i], 255.0F)) ;
  }
  vl_dsift_delete(dsift) ;
  return n ;
}
