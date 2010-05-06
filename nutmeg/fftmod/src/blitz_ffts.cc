#include <fftw3.h>
#include <list>
#include <iostream>
#ifdef THREADED
#include <pthread.h>
#endif
#ifndef NTHREADS
#define NTHREADS 1
#endif

#define INVERSE +1
#define FORWARD -1

#define MAX(a,b) ( (a) > (b) ? (a) : (b) );

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;
  
typedef struct {
  char *i;
  char *o;
  char *p;
  fftw_iodim *ft_dims;
  int ft_rank;
  fftw_iodim *nft_dims;
  int nft_rank;
  int shift;
  int direction;
} fft_args;

inline int idot(int *v1, int *v2, int N)
{
  /* Return the dot product of v1 and v2 */
  int k, p=0;
  for(k=0; k<N; k++) p += v1[k]*v2[k];
  return p;
}

inline 
int isum(int *v1, int N)
{
  /* Return the sum of the N entries in v1 */
  int k, p=0;
  for(k=0; k<N; k++) p += v1[k];
  return p;
}

inline
void indices_int(int L, int *shape, int *idc, int N)
{
  /* Given a flat index L into a ND grid with dims listed in shape,
     return the grid coordinates in idc
  */
  int i, nn=L;
  for(i=0; i<N; i++) {
    idc[N-i-1] = nn % shape[N-i-1];
    nn /= shape[N-i-1];
  }
}

void *thread_fftwf(void *ptr)
{
  fft_args *args = (fft_args *) ptr;
  fftwf_plan *FT = (fftwf_plan *) args->p;
  fftwf_complex *z_i, *z_o, *o_ptr, *i_ptr;
  float oscl, mod, shift_fix, len_xform = 1.0;
  int k, l, offsets, nxforms = 1;
  int ft_rank = args->ft_rank, nft_rank = args->nft_rank;
  int *ft_shape, *ft_strides, *ft_indices, *oft_strides;
  int *nft_shape, *nft_strides, *nft_indices, *onft_strides;

  ft_shape = new int[ft_rank];
  ft_strides = new int[ft_rank];
  ft_indices = new int[ft_rank];
  oft_strides = new int[ft_rank];

  nft_shape = new int[nft_rank];
  nft_strides = new int[nft_rank];
  nft_indices = new int[nft_rank];
  onft_strides = new int[nft_rank];

  for(k=0; k<ft_rank; k++) {
    len_xform *= args->ft_dims[k].n;
    ft_shape[k] = args->ft_dims[k].n;
    ft_strides[k] = args->ft_dims[k].is;
    oft_strides[k] = args->ft_dims[k].os;
  }
  for(k=0; k<nft_rank; k++) {
    nxforms *= args->nft_dims[k].n;
    nft_shape[k] = args->nft_dims[k].n;
    nft_strides[k] = args->nft_dims[k].is;
    onft_strides[k] = args->nft_dims[k].os;
  }
  z_i = reinterpret_cast<fftwf_complex*>( args->i );
  z_o = reinterpret_cast<fftwf_complex*>( args->o );

  // If shift is true, then len(dim_i)/2 is treated as x_i = 0,
  // and each axis grid is x_i(n) = n - len(dim_i)/2 = n + i0.
  // Therefore the term (-1)**(x_i+x_j+x_k+...) can be computed
  // as (-1)**(i+j+k+...) * (-1)**(i0+j0+k0+...)
  offsets = 0;
  for(k=0; k<ft_rank; k++) offsets += ft_shape[k]/2;
  shift_fix = offsets % 2 ? -1.0 : 1.0;

  if(args->shift) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
// 	mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
    }
  }
  fftwf_execute_dft(*FT, z_i, z_o);

  if(args->shift) {
    // if input and output pointers are different, demodulate
    // input separately without scaling on the IFFT
//     oscl = (args->direction==INVERSE) ? 1.0/len_xform : 1.0;
    oscl = (args->direction==INVERSE) ? shift_fix/len_xform : shift_fix;
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      if(args->i != args->o) {
  	o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
	for(l=0; l<(int)len_xform; l++) {
  	  indices_int(l, ft_shape, ft_indices, ft_rank);
// 	  mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	  mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	  i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
	  (*i_ptr)[0] *= mod;
	  (*i_ptr)[1] *= mod;
	}
      }
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
	mod = isum(ft_indices, ft_rank)%2 ? -oscl : oscl;
  	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
				      
    }
  } else if(args->direction == INVERSE) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
 	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] /= len_xform;
 	(*i_ptr)[1] /= len_xform;
      }
    }
  }
    


  return NULL;
}

template <const int N_rank>
void cfloat_fft(blitz::Array<cfloat,N_rank>& ai, 
		blitz::Array<cfloat,N_rank>& ao,
		int fft_rank, int *fft_dims, int direction, int shift)
{
  int n, k;
  for(k=0; k<fft_rank; k++) {
    while(fft_dims[k] < 0) fft_dims[k] += N_rank;
    if(fft_dims[k] >= N_rank) {
      std::cout<<"fft_dim is not within the rank of the array: "<<fft_dims[k]<<std::endl;
      return;
    }
  }

  fftwf_plan FT;
  // howmany_rank -- make sure it is at least 1 (with a degenerate dim if necessary)
  int howmany_rank = MAX(1, ai.rank() - fft_rank); 
  int plan_flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
  int n_threads;
  fftw_iodim *fft_iodims = new fftw_iodim[fft_rank];
  fftw_iodim *howmany_dims = new fftw_iodim[howmany_rank];
  // put down reasonable defaults for 1st non-FFT dimension
  howmany_dims[0].n = 1; howmany_dims[0].is = 1; howmany_dims[0].os = 1;
  // the fft_dims are specified in the arguments, 
  // so they are straightforward to put down
  for(k=0; k<fft_rank; k++) {
    fft_iodims[k].n = ai.shape()[ fft_dims[k] ];
    fft_iodims[k].is = ai.stride()[ fft_dims[k] ];
    fft_iodims[k].os = ao.stride()[ fft_dims[k] ];
  }
  // the "other" dims are a little more tricky..
  // 1) make a list of all the dims
  std::list<int> a_dims;
  std::list<int>::iterator it;
  for(k=0; k<ai.rank(); k++)
    a_dims.push_back(k);
  // 2) prune the fft_dims from the list
  for(k=0; k<fft_rank; k++)
    a_dims.remove(fft_dims[k]);
  // 3) put down the remaining dim info in howmany_dims
  n = 0; //howmany_rank-1;
  for(it = a_dims.begin(); it != a_dims.end(); it++) {
    howmany_dims[n].n = ai.shape()[ *it ];
    howmany_dims[n].is = ai.stride()[ *it ];
    howmany_dims[n].os = ao.stride()[ *it ];
    n++;
  }

#ifdef THREADED
  // short circuit eval I HOPE!
  // Do threads IF there are multiple FFTs AND the length of the 
  // leading non-FFT dimension is divided by NTHREADS w/o remainder
  pthread_t threads[NTHREADS];
  if( howmany_rank && !(howmany_dims[0].n % NTHREADS) ) {
    n_threads = NTHREADS;
  } else {
#endif
    n_threads = 1;
#ifdef THREADED
  }
#endif
  fft_args *args = new fft_args[n_threads];
  int block_sz = howmany_dims[0].n / n_threads;
  howmany_dims[0].n = block_sz;
  FT = fftwf_plan_guru_dft(fft_rank, fft_iodims, howmany_rank, howmany_dims,
			   reinterpret_cast<fftwf_complex*>( ai.data() ),
			   reinterpret_cast<fftwf_complex*>( ao.data() ),
			   direction, plan_flags);
  if(FT==NULL) {
    std::cout << "FFTW created a null plan, exiting" << std::endl;
    return;
  }
  

  for(n=0; n<n_threads; n++) {
    (args+n)->i = (char *) (ai.dataZero() + n*block_sz*howmany_dims[0].is);
    (args+n)->o = (char *) (ao.dataZero() + n*block_sz*howmany_dims[0].os);
    (args+n)->p = (char *) &FT;
    (args+n)->ft_dims = fft_iodims;
    (args+n)->ft_rank = fft_rank;
    (args+n)->nft_dims = howmany_dims;
    (args+n)->nft_rank = howmany_rank;
    (args+n)->shift = shift;
    (args+n)->direction = direction;
#ifdef THREADED
    pthread_create(&(threads[n]), NULL, thread_fftwf, (void *) (args+n));
#endif
  }
#ifdef THREADED
  for(n=0; n<n_threads; n++) {
    pthread_join(threads[n], NULL);
  }
#else
  void *throw_away;
  throw_away = thread_fftwf( (void *) args);
#endif
  fftwf_destroy_plan(FT);
  delete [] args;
}

void *thread_fftw(void *ptr)
{
  fft_args *args = (fft_args *) ptr;
  fftw_plan *FT = (fftw_plan *) args->p;
  fftw_complex *z_i, *z_o, *o_ptr, *i_ptr;
  double oscl, mod, shift_fix, len_xform = 1.0;
  int k, l, offsets, nxforms = 1;
  int ft_rank = args->ft_rank, nft_rank = args->nft_rank;
  int *ft_shape, *ft_strides, *ft_indices, *oft_strides;
  int *nft_shape, *nft_strides, *nft_indices, *onft_strides;

  ft_shape = new int[ft_rank];
  ft_strides = new int[ft_rank];
  ft_indices = new int[ft_rank];
  oft_strides = new int[ft_rank];

  nft_shape = new int[nft_rank];
  nft_strides = new int[nft_rank];
  nft_indices = new int[nft_rank];
  onft_strides = new int[nft_rank];

  for(k=0; k<ft_rank; k++) {
    len_xform *= args->ft_dims[k].n;
    ft_shape[k] = args->ft_dims[k].n;
    ft_strides[k] = args->ft_dims[k].is;
    oft_strides[k] = args->ft_dims[k].os;
  }
  for(k=0; k<nft_rank; k++) {
    nxforms *= args->nft_dims[k].n;
    nft_shape[k] = args->nft_dims[k].n;
    nft_strides[k] = args->nft_dims[k].is;
    onft_strides[k] = args->nft_dims[k].os;
  }
  z_i = reinterpret_cast<fftw_complex*>( args->i );
  z_o = reinterpret_cast<fftw_complex*>( args->o );

  // If shift is true, then len(dim_i)/2 is treated as x_i = 0,
  // and each axis grid is x_i(n) = n - len(dim_i)/2 = n + i0.
  // Therefore the term (-1)**(x_i+x_j+x_k+...) can be computed
  // as (-1)**(i+j+k+...) * (-1)**(i0+j0+k0+...)
  offsets = 0;
  for(k=0; k<ft_rank; k++) offsets += ft_shape[k]/2;
  shift_fix = offsets % 2 ? -1.0 : 1.0;

  if(args->shift) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
// 	mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
    }
  }
  fftw_execute_dft(*FT, z_i, z_o);

  if(args->shift) {
    // if input and output pointers are different, demodulate
    // input separately without scaling on the IFFT
//     oscl = (args->direction==INVERSE) ? 1.0/len_xform : 1.0;
    oscl = (args->direction==INVERSE) ? shift_fix/len_xform : shift_fix;
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      if(args->i != args->o) {
  	o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
	for(l=0; l<(int)len_xform; l++) {
  	  indices_int(l, ft_shape, ft_indices, ft_rank);
// 	  mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	  mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	  i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
	  (*i_ptr)[0] *= mod;
	  (*i_ptr)[1] *= mod;
	}
      }
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
	mod = isum(ft_indices, ft_rank)%2 ? -oscl : oscl;
  	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
				      
    }
  } else if(args->direction == INVERSE) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
 	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] /= len_xform;
 	(*i_ptr)[1] /= len_xform;
      }
    }
  }
    


  return NULL;
}

template <const int N_rank>
void cdouble_fft(blitz::Array<cdouble,N_rank>& ai, 
		 blitz::Array<cdouble,N_rank>& ao,
		 int fft_rank, int *fft_dims, int direction, int shift)
{
  int n, k;
  for(k=0; k<fft_rank; k++) {
    while(fft_dims[k] < 0) fft_dims[k] += N_rank;
    if(fft_dims[k] >= N_rank) {
      std::cout<<"fft_dim is not within the rank of the array: "<<fft_dims[k]<<std::endl;
      return;
    }
  }

  fftw_plan FT;
  // howmany_rank -- make sure it is at least 1 (with a degenerate dim if necessary)
  int howmany_rank = MAX(1, ai.rank() - fft_rank);
  int plan_flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
  int n_threads;
  fftw_iodim *fft_iodims = new fftw_iodim[fft_rank];
  fftw_iodim *howmany_dims = new fftw_iodim[howmany_rank];
  // put down reasonable defaults for 1st non-FFT dimension
  howmany_dims[0].n = 1; howmany_dims[0].is = 1; howmany_dims[0].os = 1;
  // the fft_dims are specified in the arguments, 
  // so they are straightforward to put down
  for(k=0; k<fft_rank; k++) {
    fft_iodims[k].n = ai.shape()[ fft_dims[k] ];
    fft_iodims[k].is = ai.stride()[ fft_dims[k] ];
    fft_iodims[k].os = ao.stride()[ fft_dims[k] ];
  }
  // the "other" dims are a little more tricky..
  // 1) make a list of all the dims
  std::list<int> a_dims;
  std::list<int>::iterator it;
  for(k=0; k<ai.rank(); k++)
    a_dims.push_back(k);
  // 2) prune the fft_dims from the list
  for(k=0; k<fft_rank; k++)
    a_dims.remove(fft_dims[k]);
  // 3) put down the remaining dim info in howmany_dims
  n = 0; //howmany_rank-1;
  for(it = a_dims.begin(); it != a_dims.end(); it++) {
    howmany_dims[n].n = ai.shape()[ *it ];
    howmany_dims[n].is = ai.stride()[ *it ];
    howmany_dims[n].os = ao.stride()[ *it ];
    n++;
  }
  
#ifdef THREADED
  // short circuit eval I HOPE!
  // Do threads IF there are multiple FFTs AND the length of the 
  // leading non-FFT dimension is divided by NTHREADS w/o remainder
  pthread_t threads[NTHREADS];
  if( howmany_rank && !(howmany_dims[0].n % NTHREADS) ) {
    n_threads = NTHREADS;
  } else {
#endif
    n_threads = 1;
#ifdef THREADED
  }
#endif
  fft_args *args = new fft_args[n_threads];
  int block_sz = howmany_dims[0].n / n_threads;
  howmany_dims[0].n = block_sz;
  FT = fftw_plan_guru_dft(fft_rank, fft_iodims, howmany_rank, howmany_dims,
			  reinterpret_cast<fftw_complex*>( ai.data() ),
			  reinterpret_cast<fftw_complex*>( ao.data() ),
			  direction, plan_flags);
  if(FT==NULL) {
    std::cout << "FFTW created a null plan, exiting" << std::endl;
    return;
  }
  
  for(n=0; n<n_threads; n++) {
    (args+n)->i = (char *) (ai.dataZero() + n*block_sz*howmany_dims[0].is);
    (args+n)->o = (char *) (ao.dataZero() + n*block_sz*howmany_dims[0].os);
    (args+n)->p = (char *) &FT;
    (args+n)->ft_dims = fft_iodims;
    (args+n)->ft_rank = fft_rank;
    (args+n)->nft_dims = howmany_dims;
    (args+n)->nft_rank = howmany_rank;
    (args+n)->shift = shift;
    (args+n)->direction = direction;
#ifdef THREADED
    pthread_create(&(threads[n]), NULL, thread_fftw, (void *) (args+n));
#endif
  }
#ifdef THREADED
  for(n=0; n<n_threads; n++) {
    pthread_join(threads[n], NULL);
  }
#else
  void *throw_away;
  throw_away = thread_fftw( (void *) args);
#endif
  fftw_destroy_plan(FT);
  delete [] args;
}

