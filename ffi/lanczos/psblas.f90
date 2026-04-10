! PSBLAS two-pass Lanczos stub.
!
! Rust FFI expects these four bind(C) symbols with exactly these signatures.
! setup() must return c_null_ptr or a valid opaque pointer.
! execute() must be idempotent (Criterion calls it hundreds of times).
! get_y() copies the result into a caller-owned buffer (double*, length len).
! teardown() must use psb_c_exit_ctxt, not psb_c_exit (avoids MPI_Finalize).
!
! All arrays from Rust are 0-based (int32 indices, double values).
! Integer kinds follow PSBLAS conventions: psb_c_ipk_ for counts/sizes.

module psblas_lanczos_stub
   use psb_base_mod
   use iso_c_binding
   use psb_cbind_const_mod
   implicit none
   private

   public :: libpsblas_lanczos_setup, libpsblas_lanczos_execute, &
      libpsblas_lanczos_get_y, libpsblas_lanczos_teardown, &
      psb_expmv_twopass

   interface psb_expmv_twopass
      module procedure psb_dexpmv_twopass
   end interface psb_expmv_twopass


contains

   ! Rust passes CSR arrays with 0-based indices (int32_t*, double*).
   ! The SpMV wrapper already calls psb_c_set_index_base(0), and checks
   ! MPI_Initialized to skip MPI_Init if already done -- same pattern here.
   ! Returns an opaque pointer that Rust stores as *mut c_void; c_null_ptr
   ! signals the benchmark harness to skip this backend.
   function libpsblas_lanczos_setup(nrows, ncols, nnz, &
      row_ptr, col_idx, values, b, krylov_dim) &
      result(ctx) bind(C, name="libpsblas_lanczos_setup")
      implicit none
      integer(psb_c_ipk_), value, intent(in) :: nrows, ncols, nnz, krylov_dim
      integer(psb_c_ipk_), intent(in) :: row_ptr(*)
      integer(psb_c_ipk_), intent(in) :: col_idx(*)
      real(c_double), intent(in) :: values(*)
      real(c_double), intent(in) :: b(*)
      type(c_ptr) :: ctx
      ctx = c_null_ptr
   end function libpsblas_lanczos_setup

   ! Criterion invokes this hundreds of times per benchmark sample.
   ! Must be idempotent: reset all working vectors at the start of each
   ! call so that repeated executions produce identical results.
   subroutine libpsblas_lanczos_execute(ctx) &
      bind(C, name="libpsblas_lanczos_execute")
      implicit none
      type(c_ptr), value, intent(in) :: ctx
   end subroutine libpsblas_lanczos_execute

   ! out is a Rust-allocated buffer (length doubles). Copy the result
   ! there; the Rust side compares it against the faer reference.
   subroutine libpsblas_lanczos_get_y(ctx, out, length) &
      bind(C, name="libpsblas_lanczos_get_y")
      implicit none
      type(c_ptr), value, intent(in) :: ctx
      real(c_double), intent(inout) :: out(*)
      integer(psb_c_ipk_), value, intent(in) :: length
   end subroutine libpsblas_lanczos_get_y

   ! Use psb_c_exit_ctxt, not psb_c_exit: the latter calls MPI_Finalize
   ! and Criterion runs multiple benchmarks in the same process.
   subroutine libpsblas_lanczos_teardown(ctx) &
      bind(C, name="libpsblas_lanczos_teardown")
      implicit none
      type(c_ptr), value, intent(in) :: ctx
   end subroutine libpsblas_lanczos_teardown

   subroutine psb_dexpmv_twopass(a,desc_a,b,x,tol,maxit,info,itrace)
      ! Implementation of the two-pass Lanczos algorithm for computing
      ! the matrix exponential times a vector
      use psb_base_mod
      implicit none
      type(psb_dspmat_type), intent(in) :: a
      type(psb_desc_type), intent(in) :: desc_a
      type(psb_d_vect_type), intent(in) :: b
      type(psb_d_vect_type), intent(inout) :: x
      real(psb_dpk_), intent(in) :: tol
      integer(psb_ipk_), intent(in) :: maxit
      integer(psb_ipk_), intent(out) :: info
      integer(psb_ipk_), intent(in), optional :: itrace


      ! Local variables
      integer(psb_ipk_) :: me, np
      integer(psb_lkp_) :: mglob
      integer(psb_ipk_) :: n_row, n_col
      integer(psb_ipk_) :: i, itwo_pass
      real(psb_dpk_) :: beta, beta0, err

      type(psb_cxtxt_type) :: ctxt
      ! To store the tridiagonal matrix for the Krylov subspace projection
      ! we just need two vectors of length maxit, and maxit - 1 for the alphas
      ! and the betas
      real(psb_dpk_), allocatable, dimension(:) :: alphas, betas
      ! since we are going to use BLAS/LAPACK to compute the matrix exponential
      ! of the tridiagonal matrix, we need also a copy of this because
      ! calls to Lapack will overwrite the input matrix
      real(psb_dpk_), allocatable, dimension(:) :: alphas_wrk, betas_wrk
      type(psb_d_vect_type), allocatable, target :: wwrk(:)
      type(psb_d_vect_type), pointer :: wj, vj, vjm1
      ! Work array for the dsteqr call to compute the eigenvalues
      ! and eigenvectors of the tridiagonal matrix, the uvec(:) will
      ! store the result of exp(T_m) e1, where T_m is the tridiagonal matrix
      real(psb_dpk_), allocatable :: dsteqr_work(:), uvec(:)
      real(psb_dpk_), allocatable, dimension(:,:) :: Q

      ! Shadow variables for optional arguments
      integer(psb_ipk_) :: itrace_

      ! Error handling variables
      integer(psb_ipk_) :: debug_level, debug_unit, err_act
      character(len=20) :: name
      character(len=20) :: methdname

      ! Initialize error handling parameters
      info = psb_success_
      name = "psb_dexpmv_twopass"
      call psb_erractionsave(err_act)
      debug_unit = psb_get_debug_unit()
      debug_level = psb_get_debug_level()

      ! Get the PSBLAS context and discover who we are
      ctxt = desc_a%get_context()
      call psb_info(ctxt, me, np)

      ! Check that the stuff we are given is valid
      if (.not.allocated(b%v)) then
         info = psb_err_invalid_vect_state_
         call psb_errpush(info, name)
         goto 9999
      end if
      if (.not.allocated(x%v)) then
         info = psb_err_invalid_vect_state_
         call psb_errpush(info, name)
         goto 9999
      end if

      ! Get the global matrix dimensions
      mglob = desc_a%get_global_rows()
      ! And the local ones
      n_row = desc_a%get_local_rows()
      n_col = desc_a%get_global_cols()

      ! We use them to check the state of the input vectors
      call psb_chkvect(mglob,lone,x%get_nrows(),lone,lone,desc_a,info)
      if (info == psb_success_)&
      & call psb_chkvect(mglob,lone,b%get_nrows(),lone,lone,desc_a,info)
      if(info /= psb_success_) then
         info=psb_err_from_subroutine_
         call psb_errpush(info,name,a_err='psb_chkvect on X/B')
         goto 9999
      end if

      ! Now everything is valid, and we can allocate the work vectors
      if (info == psb_success_) call psb_geall(wwrk,desc_a,info,n=3_psb_ipk_)
      if (info == psb_success_) call psb_geasb(wwrk,desc_a,info,mold=x%v,scratch=.true.)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      ! We allocate the tridiagonal matrix for the Krylov subspace projection
      if (info == psb_success_) then
         allocate(alphas(maxit), betas(maxit), alphas_wrk(maxit), betas_wrk(maxit-1), stat=info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end if
      ! We allocate the uvec for the result of exp(T_m) e1, where T_m is the tridiagonal matrix
      if (info == psb_success_) then
         allocate(uvec(maxit), stat=info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end if

      ! To compute the dense matrix exponential we use dsteqr from LAPACK,
      ! this needs a work array, since we perform multiple calls, we allocate
      ! the work array once and reuse it, we compute the size of the work array
      ! as 2*maxit - 2, which is the size required by dsteqr for the tridiagonal case
      allocate(dsteqr_work(2*maxit - 2), stat=info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      allocate(Q(maxit, maxit), stat=info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if

      ! Set up pointers to the work vectors for easier access
      wj => wwrk(1)
      vj => wwrk(2)
      vjm1 => wwrk(3)

      ! Shadow the optional arguments
      if (present(itrace)) then
         itrace_ = itrace
      else
         itrace_ = 0_psb_ipk_
      end if

      ! Main Lanczos iteration loop:
      beta0 = psb_norm2(b, desc_a, info)  ! beta_0 = ||b||_2, saved for final scaling
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      beta = beta0
      call psb_geaxpy(1.0_psb_dpk_/beta0, b, dzero, vj, desc_a, info) ! vj = b / ||b||
      firstlanczos: do i = 1, maxit-1
         !wj = A * vj - beta(i) * vjm1 = A * vj
         if (i == 1) then
            call psb_gemv('N', 1.0_psb_dpk_, a, vj, 0.0_psb_dpk_, wj, desc_a, info)
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
         else
            call psb_gemv('N', 1.0_psb_dpk_, a, vj, 0.0_psb_dpk_, wj, desc_a, info)
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
            call psb_geaxpy(-betas(i), vjm1, 1.0_psb_dpk_, wj, desc_a, info) ! wj -= beta_i * v_{j-1}
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
         end if
         ! alpha(i) = vj^T * wj = tridiag(i, i)
         alphas(i) = psb_gedot(vj, wj, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         ! wj = wj - alpha(i)  * vj = wj - tridiag(i, i) * vj
         call psb_geaxpy(-alphas(i), vj, 1.0_psb_dpk_, wj, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         ! betas(i+1) = ||wj||_2 = tridiag(i+1, i)
         beta = psb_norm2(wj, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         betas(i+1) = beta
         ! Check for convergence or breakdown
         if (beta == dzero) then
            ! Lanczos breakdown: stop the iteration
            if (debug_level >= psb_debug_ext_) then
               write(debug_unit, *) me, ' ', trim(name), ': Lanczos breakdown at iteration ', i
            end if
            exit firstlanczos
         end if
         ! Compute the error estimatore for the matrix exponential
         ! || r_{m} ||_2 = ||b|| * |beta_{m+1}| * |e_m^T exp(T_m) e_1|
         alphas_wrk(1:i)  = alphas(1:i)
         betas_wrk(1:i-1) = betas(2:i)   ! subdiagonal is beta_2,...,beta_i stored at betas(2:i)
         call expm_tridiag_core(i, alphas_wrk(1:i), betas_wrk(1:i-1), betas(i+1), beta0, Q, dsteqr_work, uvec, err)
         ! Check if the itrace_ > 0 and if mod(itrace_, i) == 0, if so print the error estimator
         if (itrace_ > 0 .and. mod(i, itrace_) == 0) then
            write(psb_output_unit, '(A, I5, A, ES12.5)') 'Iteration ', i, ': error estimator = ', err
         end if
         if (err < tol) then
            if (itrace_ > 0) then
               write(psb_output_unit, '(A, I5, A, ES12.5)') 'Converged at iteration ', i, ': error estimator = ', err
            end if
            exit firstlanczos
         end if
         ! Prepare for the next iteration: vjm1 = vj, vj = wj / beta
         call psb_geaxpy(1.0_psb_dpk_, vj, 0.0_psb_dpk_, vjm1, desc_a, info) ! vjm1 = vj
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         call psb_geaxpy(1.0_psb_dpk_/beta, wj, 0.0_psb_dpk_, vj, desc_a, info) ! vj = wj / beta
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end do

      ! After the Lanczos iteration, we have the tridiagonal matrix defined by alphas and betas,
      ! and we have the computed linear coefficients for the krylov basis in
      ! uvec, we would like to compute the final result x = ||b|| * V_m * exp(T_m) e1 = ||b|| * V_m * uvec,
      ! where V_m is the matrix with columns v1, v2, ..., vm, but we have not stored
      ! the krylov basis vectors, we have only the last two, so we need to reconstruct the krylov basis
      ! vectors from the Lanczos iteration, we can do this by performing a second pass of the Lanczos iteration,
      ! but this time we will compute x <- x + u(itwo_pass) * vj, where u(itwo_pass) is the
      ! itwo_pass-th element of the uvec, and vj is the itwo_pass-th krylov basis vector.
      ! Observe also that for the new pass we don't need to compute the tridiagonal matrix entries
      ! because we already have them, we just need to perform the matrix-vector products and
      ! the axpy operations to reconstruct the krylov basis vectors, and then compute the final result.
      ! Zero x before accumulating, then restart Lanczos from v1 = b/||b||, vjm1 = 0
      call psb_geaxpy(dzero, vj, dzero, x, desc_a, info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      call psb_geaxpy(1.0_psb_dpk_/beta0, b, dzero, vj, desc_a, info)  ! vj = b / ||b||
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      call psb_geaxpy(dzero, vj, dzero, vjm1, desc_a, info)              ! vjm1 = 0
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      do itwo_pass = 1,i
         if (itwo_pass == 1) then
            call psb_gemv('N', 1.0_psb_dpk_, a, vj, 0.0_psb_dpk_, wj, desc_a, info)
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
         else
            call psb_gemv('N', 1.0_psb_dpk_, a, vj, 0.0_psb_dpk_, wj, desc_a, info)
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
            call psb_geaxpy(-betas(itwo_pass), vjm1, 1.0_psb_dpk_, wj, desc_a, info) ! wj -= beta_j * v_{j-1}
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
         end if
         ! x = x + uvec(itwo_pass) * vj
         call psb_geaxpy(uvec(itwo_pass), vj, 1.0_psb_dpk_, x, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         ! Prepare for the next iteration: vjm1 = vj, vj = wj / beta
         ! Advance Lanczos vectors for the next step (skipped on last iteration)
         if (itwo_pass < i) then
            call psb_geaxpy(1.0_psb_dpk_, vj, dzero, vjm1, desc_a, info) ! vjm1 = vj
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
            call psb_geaxpy(1.0_psb_dpk_/betas(itwo_pass+1), wj, dzero, vj, desc_a, info) ! vj = wj / beta_{j+1}
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
         end if
      end do
      ! Scale x by ||b|| = beta0 to obtain the final result
      call psb_gescal(beta0, x, desc_a, info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if

      ! And we are done, we free work arrays and restore error handling state before returning
      if (allocated(wwrk)) deallocate(wwrk, stat=info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      ! Deallocate the small arrays for the tridiagonal matrix
      if (allocated(alphas)) deallocate(alphas, stat=info)
      if (allocated(betas)) deallocate(betas, stat=info)
      if (allocated(alphas_wrk)) deallocate(alphas_wrk, stat=info)
      if (allocated(betas_wrk)) deallocate(betas_wrk, stat=info)
      if (allocated(dsteqr_work)) deallocate(dsteqr_work, stat=info)
      if (allocated(Q)) deallocate(Q, stat=info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if

      call psb_erractionrestore(err_act)
      return

9999  call psb_error_handler(err_act)
      return

   end subroutine psb_dexpmv_twopass

   ! Implementation of the dense matrix exponential for the tridiagonal matrix
   ! in the Krylov subspace projection, we make things simple and use BLAS/LAPACK
   ! to compute the eigenvalues and eigenvectors of the tridiagonal matrix,
   ! and then compute the matrix exponential of the tridiagonal matrix using the eigen-decomposition.
   subroutine expm_tridiag_core(m, d, e, beta_m, vnorm, Q, work, u, err)
      ! Computes:
      !   u   = exp(T_m) e1
      !   err = ||v|| * |beta_m| * |e_m^T u|
      !
      ! INPUT:
      !   m        : size of T_m
      !   d(m)     : diagonal (overwritten with eigenvalues)
      !   e(m-1)   : subdiagonal (destroyed)
      !   beta_m   : last Lanczos coefficient
      !   vnorm    : ||v||
      !
      ! WORKSPACE (provided by caller):
      !   Q(m,m)       : eigenvectors
      !   work(2*m-2)  : LAPACK workspace
      !
      ! OUTPUT:
      !   u(m)     : exp(T_m) e1
      !   err      : error estimate

      integer, intent(in) :: m
      double precision, intent(inout) :: d(m)
      double precision, intent(inout) :: e(m-1)
      double precision, intent(in) :: beta_m, vnorm
      double precision, intent(out) :: Q(m,m)
      double precision, intent(inout) :: work(*)
      double precision, intent(out) :: u(m)
      double precision, intent(out) :: err

      double precision :: w(m)
      integer :: i, info

      ! Initialize Q = I
      Q = 0.0d0
      do i = 1, m
         Q(i,i) = 1.0d0
      end do

      ! Eigendecomposition T_m = Q Λ Q^T
      call dsteqr('I', m, d, e, Q, m, work, info)
      if (info /= 0) then
         print *, "Error in DSTEQR, info=", info
         stop
      end if

      ! w = Q^T e1 = first row of Q
      do i = 1, m
         w(i) = Q(1, i)
      end do

      ! Apply exponential of eigenvalues
      do i = 1, m
         w(i) = exp(d(i)) * w(i)
      end do

      ! u = Q * w = exp(T_m) e1
      call dgemv('N', m, m, 1.0d0, Q, m, w, 1, 0.0d0, u, 1)

      ! Error estimator: ||v|| * |beta_m| * |e_m^T u|
      err = vnorm * abs(beta_m) * abs(u(m))

   end subroutine expm_tridiag_core

end module psblas_lanczos_stub
