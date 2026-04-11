! PSBLAS one-pass Lanczos stub for f(A)b.
!
! Rust FFI expects these four bind(C) symbols with exactly these signatures,
! matching the libpsblas_lanczos_two_pass_* family modulo the name prefix.
! setup() must return c_null_ptr or a valid opaque pointer.
! get_y() copies the result into a caller-owned buffer (double*, length len).
! teardown() must use psb_c_exit_ctxt, not psb_c_exit (avoids MPI_Finalize).
!
! All arrays from Rust are 0-based (int32 indices, double values).
! Integer kinds follow PSBLAS conventions: psb_c_ipk_ for counts/sizes.
!
! Every O(n) and O(n*k) buffer belongs to the context allocated in
! setup; execute reuses them without any fresh allocation. Criterion's
! timing window measures execute in isolation, matching the faer
! lanczos_into / lanczos_two_pass_into allocation policy.

module psblas_lanczos
   use iso_c_binding
   use psb_cbind_const_mod
   implicit none
contains

   ! Rust passes CSR arrays with 0-based indices (int32_t*, double*).
   ! The SpMV wrapper already calls psb_c_set_index_base(0), and checks
   ! MPI_Initialized to skip MPI_Init if already done, same pattern here.
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

end module psblas_lanczos
