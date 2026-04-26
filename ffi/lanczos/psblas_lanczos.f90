! PSBLAS one-pass Lanczos for f(A)b.

module psblas_lanczos
   use psb_base_mod
   use iso_c_binding
   use psb_objhandle_mod

   implicit none
   private

   public :: psb_expmv_onepass

   interface psb_expmv_onepass
      module procedure psb_dexpmv_onepass
   end interface psb_expmv_onepass

contains

   function psb_c_dexpmv_onepass(ah,desc_ah,bh,xh,tol,maxit) bind(C) result(info)
      use iso_c_binding
      use psb_cbind_const_mod
      implicit none
      type(psb_c_descriptor) :: desc_ah
      type(psb_c_dvector) :: bh, xh
      type(psb_c_dspmat) :: ah
      real(c_double), value :: tol
      integer(c_int), value :: maxit
      integer(c_int) :: info

      type(psb_dspmat_type), pointer :: a
      type(psb_desc_type), pointer :: desc_a
      type(psb_d_vect_type), pointer :: b, x
      real(psb_dpk_) :: tol_f
      integer(psb_ipk_) :: maxit_f, info_f

      tol_f = tol
      maxit_f = maxit
      if (c_associated(desc_ah%item)) then
         call c_f_pointer(desc_ah%item, desc_a)
      else
         info = -1
         return
      end if
      if (c_associated(ah%item)) then
         call c_f_pointer(ah%item, a)
      else
         info = -1
         return
      end if
      if (c_associated(bh%item)) then
         call c_f_pointer(bh%item, b)
      else
         info = -1
         return
      end if
      if (c_associated(xh%item)) then
         call c_f_pointer(xh%item, x)
      else
         info = -1
         return
      end if

      call psb_dexpmv_onepass(a, desc_a, b, x, tol_f, maxit_f, info_f)

      info = info_f

   end function psb_c_dexpmv_onepass

   subroutine psb_dexpmv_onepass(a,desc_a,b,x,tol,maxit,info,itrace)
      use psb_base_mod
      implicit none
      type(psb_dspmat_type), intent(in) :: a
      type(psb_desc_type), intent(in) :: desc_a
      type(psb_d_vect_type), intent(inout) :: b
      type(psb_d_vect_type), intent(inout) :: x
      real(psb_dpk_), intent(in) :: tol
      integer(psb_ipk_), intent(in) :: maxit
      integer(psb_ipk_), intent(out) :: info
      integer(psb_ipk_), intent(in), optional :: itrace

      integer(psb_ipk_) :: me, np
      integer(psb_lpk_) :: mglob
      integer(psb_ipk_) :: i, j
      real(psb_dpk_) :: beta, beta0, err

      type(psb_ctxt_type) :: ctxt
      real(psb_dpk_), allocatable, dimension(:) :: alphas, betas
      real(psb_dpk_), allocatable, dimension(:) :: alphas_wrk, betas_wrk
      ! wwrk(1..maxit) hold the Krylov basis V_m; wwrk(maxit+1) is the wj scratch.
      type(psb_d_vect_type), allocatable, target :: wwrk(:)
      type(psb_d_vect_type), pointer :: wj, vj, vjm1
      real(psb_dpk_), allocatable :: dsteqr_work(:), uvec(:)
      real(psb_dpk_), allocatable, dimension(:,:) :: Q

      integer(psb_ipk_) :: itrace_

      integer(psb_ipk_) :: debug_level, debug_unit, err_act
      character(len=20) :: name

      info = psb_success_
      name = "psb_dexpmv_onepass"
      call psb_erractionsave(err_act)
      debug_unit = psb_get_debug_unit()
      debug_level = psb_get_debug_level()

      ctxt = desc_a%get_context()
      call psb_info(ctxt, me, np)

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

      mglob = desc_a%get_global_rows()

      call psb_chkvect(mglob,lone,x%get_nrows(),lone,lone,desc_a,info)
      if (info == psb_success_)&
      & call psb_chkvect(mglob,lone,b%get_nrows(),lone,lone,desc_a,info)
      if(info /= psb_success_) then
         info=psb_err_from_subroutine_
         call psb_errpush(info,name,a_err='psb_chkvect on X/B')
         goto 9999
      end if

      if (info == psb_success_) call psb_geall(wwrk,desc_a,info,n=maxit+1_psb_ipk_)
      if (info == psb_success_) call psb_geasb(wwrk,desc_a,info,mold=x%v,scratch=.true.)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      if (info == psb_success_) then
         allocate(alphas(maxit), betas(maxit), alphas_wrk(maxit), betas_wrk(maxit-1), stat=info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end if
      if (info == psb_success_) then
         allocate(uvec(maxit), stat=info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end if

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

      wj => wwrk(maxit+1)

      if (present(itrace)) then
         itrace_ = itrace
      else
         itrace_ = 0_psb_ipk_
      end if

      beta0 = psb_norm2(b, desc_a, info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      beta = beta0
      ! V_basis(1) = b / ||b||
      call psb_geaxpby(1.0_psb_dpk_/beta0, b, dzero, wwrk(1), desc_a, info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if

      lanczos_iter: do i = 1, maxit-1
         vj => wwrk(i)
         call psb_spmm(1.0_psb_dpk_, a, vj, 0.0_psb_dpk_, wj, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         if (i > 1) then
            vjm1 => wwrk(i-1)
            call psb_geaxpby(-betas(i), vjm1, 1.0_psb_dpk_, wj, desc_a, info) ! wj -= beta_i * v_{i-1}
            if (info /= psb_success_) then
               info=psb_err_from_subroutine_non_
               call psb_errpush(info,name)
               goto 9999
            end if
         end if
         alphas(i) = psb_gedot(vj, wj, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         call psb_geaxpby(-alphas(i), vj, 1.0_psb_dpk_, wj, desc_a, info) ! wj -= alpha_i * v_i
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         beta = psb_norm2(wj, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
         betas(i+1) = beta
         if (beta <= tiny(beta)) then
            if (debug_level >= psb_debug_ext_) then
               write(debug_unit, *) me, ' ', trim(name), ': Lanczos breakdown at iteration ', i
            end if
            exit lanczos_iter
         end if
         alphas_wrk(1:i)  = alphas(1:i)
         betas_wrk(1:i-1) = betas(2:i)
         call expm_tridiag_core(i, alphas_wrk(1:i), betas_wrk(1:i-1), betas(i+1), beta0, Q, dsteqr_work, uvec, err)
         if (itrace_ > 0 .and. mod(i, itrace_) == 0) then
            write(psb_out_unit, '(A, I5, A, ES12.5)') 'Iteration ', i, ': error estimator = ', err
         end if
         if (err < tol) then
            if (itrace_ > 0) then
               write(psb_out_unit, '(A, I5, A, ES12.5)') 'Converged at iteration ', i, ': error estimator = ', err
            end if
            exit lanczos_iter
         end if
         ! V_basis(i+1) = wj / beta_{i+1}
         call psb_geaxpby(1.0_psb_dpk_/beta, wj, dzero, wwrk(i+1), desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end do lanczos_iter
      i = min(i, maxit-1)

      ! x = sum_{j=1..m} (beta0 * uvec(j)) * V_basis(j)
      call psb_geaxpby(dzero, wwrk(1), dzero, x, desc_a, info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      do j = 1, i
         call psb_geaxpby(beta0 * uvec(j), wwrk(j), 1.0_psb_dpk_, x, desc_a, info)
         if (info /= psb_success_) then
            info=psb_err_from_subroutine_non_
            call psb_errpush(info,name)
            goto 9999
         end if
      end do

      if (allocated(wwrk)) deallocate(wwrk, stat=info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if
      if (allocated(alphas)) deallocate(alphas, stat=info)
      if (allocated(betas)) deallocate(betas, stat=info)
      if (allocated(alphas_wrk)) deallocate(alphas_wrk, stat=info)
      if (allocated(betas_wrk)) deallocate(betas_wrk, stat=info)
      if (allocated(dsteqr_work)) deallocate(dsteqr_work, stat=info)
      if (allocated(Q)) deallocate(Q, stat=info)
      if (allocated(uvec)) deallocate(uvec, stat=info)
      if (info /= psb_success_) then
         info=psb_err_from_subroutine_non_
         call psb_errpush(info,name)
         goto 9999
      end if

      call psb_erractionrestore(err_act)
      return

9999  call psb_error_handler(err_act)
      return

   end subroutine psb_dexpmv_onepass

   subroutine expm_tridiag_core(m, d, e, beta_m, vnorm, Q, work, u, err)
      use psb_base_mod
      implicit none
      integer, intent(in) :: m
      real(psb_dpk_), intent(inout) :: d(m)
      real(psb_dpk_), intent(inout) :: e(m-1)
      real(psb_dpk_), intent(in) :: beta_m, vnorm
      real(psb_dpk_), intent(out) :: Q(m,m)
      real(psb_dpk_), intent(inout) :: work(*)
      real(psb_dpk_), intent(out) :: u(m)
      real(psb_dpk_), intent(out) :: err

      real(psb_dpk_) :: w(m)
      integer(psb_ipk_) :: i, info

      Q = 0.0_psb_dpk_
      do i = 1, m
         Q(i,i) = 1.0_psb_dpk_
      end do

      call dsteqr('I', m, d, e, Q, m, work, info)
      if (info /= 0) then
         write(psb_out_unit,'(A,I0)') "Error in DSTEQR, info=", info
         stop
      end if

      do i = 1, m
         w(i) = Q(1, i)
      end do

      do i = 1, m
         w(i) = exp(-d(i)) * w(i)
      end do

      call dgemv('N', m, m, 1.0_psb_dpk_, Q, m, w, 1, 0.0_psb_dpk_, u, 1)

      err = vnorm * abs(beta_m) * abs(u(m))

   end subroutine expm_tridiag_core

end module psblas_lanczos
