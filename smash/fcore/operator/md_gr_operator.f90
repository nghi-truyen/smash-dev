!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr_interception
!%      - gr_production
!%      - gr_exchange
!%      - gr_threshold_exchange
!%      - gr_transfer
!%      - gr_production_transfer_mlp_alg
!%      - gr_production_transfer_ode
!%      - gr_production_transfer_mlp_ode
!%      - gr4_time_step
!%      - gr4_mlp_alg_time_step
!%      - gr4_ode_time_step
!%      - gr4_mlp_ode_time_step
!%      - gr5_time_step
!%      - grd_time_step
!%      - loieau_time_step

module md_gr_operator

    use md_constant !% only : sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_options !% only: OptionsDT
    use mwd_atmos_manipulation !% get_ac_atmos_data_time_step
    use mwd_nn_parameters !% only: NN_Parameters_LayerDT
    use md_algebra !% only: solve_linear_system_2vars
    use md_neural_network !% only: forward_mlp

    implicit none

contains

    subroutine gr_interception(prcp, pet, ci, hi, pn, en)

        implicit none

        real(sp), intent(in) :: prcp, pet, ci
        real(sp), intent(inout) :: hi
        real(sp), intent(out) :: pn, en

        real(sp) :: ei

        ei = min(pet, prcp + hi*ci)

        pn = max(0._sp, prcp - ci*(1._sp - hi) - ei)

        en = pet - ei

        hi = hi + (prcp - ei - pn)/ci

    end subroutine gr_interception

    subroutine gr_production(pn, en, cp, beta, hp, pr, perc)

        implicit none

        real(sp), intent(in) :: pn, en, cp, beta
        real(sp), intent(inout) :: hp
        real(sp), intent(out) :: pr, perc

        real(sp) :: inv_cp, ps, es, hp_imd

        inv_cp = 1._sp/cp
        pr = 0._sp

        ps = cp*(1._sp - hp*hp)*tanh(pn*inv_cp)/ &
        & (1._sp + hp*tanh(pn*inv_cp))

        es = (hp*cp)*(2._sp - hp)*tanh(en*inv_cp)/ &
        & (1._sp + (1._sp - hp)*tanh(en*inv_cp))

        hp_imd = hp + (ps - es)*inv_cp

        if (pn .gt. 0) then

            pr = pn - (hp_imd - hp)*cp

        end if

        perc = (hp_imd*cp)*(1._sp - (1._sp + (hp_imd/beta)**4)**(-0.25_sp))

        hp = hp_imd - perc*inv_cp

    end subroutine gr_production

    subroutine gr_exchange(kexc, ht, l)

        implicit none

        real(sp), intent(in) :: kexc
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: l

        l = kexc*ht**3.5_sp

    end subroutine gr_exchange

    subroutine gr_threshold_exchange(kexc, aexc, ht, l)

        implicit none

        real(sp), intent(in) :: kexc, aexc
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: l

        l = kexc*(ht - aexc)

    end subroutine gr_threshold_exchange

    subroutine gr_transfer(n, prcp, pr, ct, ht, q)

        implicit none

        real(sp), intent(in) :: n, prcp, pr, ct
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: q

        real(sp) :: pr_imd, ht_imd, nm1, d1pnm1

        nm1 = n - 1._sp
        d1pnm1 = 1._sp/nm1

        if (prcp .lt. 0._sp) then

            pr_imd = ((ht*ct)**(-nm1) - ct**(-nm1))**(-d1pnm1) - (ht*ct)

        else

            pr_imd = pr

        end if

        ht_imd = max(1.e-6_sp, ht + pr_imd/ct)

        ht = (((ht_imd*ct)**(-nm1) + ct**(-nm1))**(-d1pnm1))/ct

        q = (ht_imd - ht)*ct

    end subroutine gr_transfer

    subroutine gr_production_transfer_mlp_alg(layers, neurons, pn, en, cp, beta, ct, kexc, n, prcp, hp, ht, q)
        !% Integrate MLP in stepwise aprroximation method

        implicit none

        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), intent(in) :: pn, en, cp, beta, ct, kexc, n, prcp
        real(sp), intent(inout) :: hp, ht, q

        real(sp), dimension(4) :: input_layer  ! fixed NN input size
        real(sp), dimension(5) :: output_layer  ! fixed NN output size
        real(sp) :: pr, perc, ps, es, hp_imd, pr_imd, ht_imd, nm1, d1pnm1, qd

        input_layer(1) = hp
        input_layer(2) = ht
        input_layer(3) = pn
        input_layer(4) = en

        call forward_mlp(layers, neurons, input_layer, output_layer)

        pr = 0._sp

        ps = cp*(1._sp - hp*hp)*tanh(pn/cp)/(1._sp + hp*tanh(pn/cp))

        es = (hp*cp)*(2._sp - hp)*tanh(en/cp)/(1._sp + (1._sp - hp)*tanh(en/cp))

        hp_imd = hp + (output_layer(1)*ps - output_layer(2)*es)/cp

        if (pn .gt. 0) then

            pr = pn - (hp_imd - hp)*cp

        end if

        perc = (hp_imd*cp)*(1._sp - (1._sp + (hp_imd/beta)**4)**(-0.25_sp))

        hp = hp_imd - perc/cp

        nm1 = n - 1._sp
        d1pnm1 = 1._sp/nm1

        if (prcp .lt. 0._sp) then

            pr_imd = ((ht*ct)**(-nm1) - ct**(-nm1))**(-d1pnm1) - (ht*ct)

        else

            pr_imd = 0.9_sp*(output_layer(3)*pr + output_layer(4)*perc) &
            & + output_layer(5)*kexc*(ht**3.5_sp)

        end if

        ht_imd = max(1.e-6_sp, ht + pr_imd/ct)

        ht = (((ht_imd*ct)**(-nm1) + ct**(-nm1))**(-d1pnm1))/ct

        qd = 0.1_sp*(output_layer(3)*pr + output_layer(4)*perc) + output_layer(5)*kexc*(ht**3.5_sp)

        q = (ht_imd - ht)*ct + max(0._sp, qd)

    end subroutine gr_production_transfer_mlp_alg

    subroutine gr_production_transfer_ode(pn, en, cp, ct, kexc, hp, ht, q)
        !% Solve state-space ODE system with implicit Euler

        implicit none

        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, q

        real(sp), dimension(2, 2) :: jacob
        real(sp), dimension(2) :: dh, delta_h
        real(sp) :: hp0, ht0, dt, fhp, fht, tmp_j
        integer :: i, j
        integer :: n_subtimesteps = 2
        integer :: maxiter = 10

        dt = 1._sp/real(n_subtimesteps, sp)

        do i = 1, n_subtimesteps

            hp0 = hp
            ht0 = ht

            do j = 1, maxiter

                fhp = (1._sp - hp**2)*pn - hp*(2._sp - hp)*en
                dh(1) = hp - hp0 - dt*fhp/cp

                fht = ct*ht**5 - 0.9_sp*pn*hp**2 - kexc*ht**3.5_sp
                dh(2) = ht - ht0 + dt*fht/ct  ! fht here is -fht

                jacob(1, 1) = 1._sp + dt*2._sp*(hp*(pn - en) + en)/cp

                jacob(1, 2) = 0._sp

                jacob(2, 1) = dt*1.8_sp*pn*hp/ct

                tmp_j = 5._sp*ht**4 - 3.5_sp*kexc*(ht**2.5_sp)/ct
                jacob(2, 2) = 1._sp + dt*tmp_j

                call solve_linear_system_2vars(jacob, delta_h, dh)

                hp = hp + delta_h(1)
                if (hp .le. 0._sp) hp = 1.e-6_sp
                if (hp .ge. 1._sp) hp = 1._sp - 1.e-6_sp

                ht = ht + delta_h(2)
                if (ht .le. 0._sp) ht = 1.e-6_sp
                if (ht .ge. 1._sp) ht = 1._sp - 1.e-6_sp

                if (sqrt((delta_h(1)/hp)**2 + (delta_h(2)/ht)**2) .lt. 1.e-6_sp) exit

            end do

        end do

        q = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_production_transfer_ode

    subroutine gr_production_transfer_mlp_ode(layers, neurons, pn, en, cp, ct, kexc, hp, ht, q)
        !% Solve state-space ODE system with explicit Euler and MLP

        implicit none

        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, q

        real(sp), dimension(4) :: input_layer  ! fixed NN input size
        real(sp), dimension(5) :: output_layer  ! fixed NN output size
        real(sp) :: dt, fhp, fht
        integer :: i
        integer :: n_subtimesteps = 4

        input_layer(1) = hp
        input_layer(2) = ht
        input_layer(3) = pn
        input_layer(4) = en

        call forward_mlp(layers, neurons, input_layer, output_layer)

        dt = 1._sp/real(n_subtimesteps, sp)

        do i = 1, n_subtimesteps

            fhp = output_layer(1)*pn*(1._sp - hp**2) - output_layer(2)*en*hp*(2._sp - hp)
            hp = hp + dt*fhp/cp
            if (hp .le. 0._sp) hp = 1.e-6_sp
            if (hp .ge. 1._sp) hp = 1._sp - 1.e-6_sp

            fht = output_layer(3)*0.9_sp*pn*hp**2 - output_layer(4)*ct*ht**5 + &
            & output_layer(5)*kexc*ht**3.5_sp
            ht = ht + dt*fht/ct
            if (ht .le. 0._sp) ht = 1.e-6_sp
            if (ht .ge. 1._sp) ht = 1._sp - 1.e-6_sp

        end do

        q = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_production_transfer_mlp_ode

    subroutine gr4_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: pn, en, pr, perc, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, &
        !$OMP& ac_qt) &
        !$OMP& private(row, col, k, pn, en, pr, perc, l, prr, prd, qr, qd)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(pn, en, ac_cp(k), 9._sp/4._sp, ac_hp(k), pr, perc)

                    call gr_exchange(ac_kexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr4_time_step

    subroutine gr4_mlp_alg_time_step(setup, mesh, input_data, options, time_step, layers, neurons, &
    & ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: pn, en

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, layers, neurons, ac_prcp, ac_pet, ac_ci, ac_cp, ac_ct, ac_kexc, &
        !$OMP& ac_hi, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, pn, en)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                else

                    pn = 0._sp
                    en = 0._sp

                end if

                call gr_production_transfer_mlp_alg(layers, neurons, pn, en, &
                & ac_cp(k), 9._sp/4._sp, ac_ct(k), ac_kexc(k), 5._sp, &
                & ac_prcp(k), ac_hp(k), ac_ht(k), ac_qt(k))

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr4_mlp_alg_time_step

    subroutine gr4_ode_time_step(setup, mesh, input_data, options, time_step, &
    & ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: pn, en

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, &
        !$OMP& ac_qt) &
        !$OMP& private(row, col, k, pn, en)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                else

                    pn = 0._sp
                    en = 0._sp

                end if

                call gr_production_transfer_ode(pn, en, ac_cp(k), ac_ct(k), &
                & ac_kexc(k), ac_hp(k), ac_ht(k), ac_qt(k))

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr4_ode_time_step

    subroutine gr4_mlp_ode_time_step(setup, mesh, input_data, options, time_step, layers, neurons, &
    & ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: pn, en

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, layers, neurons, ac_prcp, ac_pet, ac_ci, ac_cp, ac_ct, ac_kexc, &
        !$OMP& ac_hi, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, pn, en)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                else

                    pn = 0._sp
                    en = 0._sp

                end if

                call gr_production_transfer_mlp_ode(layers, neurons, pn, en, ac_cp(k), &
                & ac_ct(k), ac_kexc(k), ac_hp(k), ac_ht(k), ac_qt(k))

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr4_mlp_ode_time_step

    subroutine gr5_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_kexc, ac_aexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc, ac_aexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: pn, en, pr, perc, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, ac_ct, ac_kexc, ac_aexc, ac_hi, ac_hp, &
        !$OMP& ac_ht, ac_qt) &
        !$OMP& private(row, col, k, pn, en, pr, perc, l, prr, prd, qr, qd)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(pn, en, ac_cp(k), 9._sp/4._sp, ac_hp(k), pr, perc)

                    call gr_threshold_exchange(ac_kexc(k), ac_aexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr5_time_step

    subroutine grd_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_cp, ac_ct, ac_hp, &
    & ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_cp, ac_ct
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: ei, pn, en, pr, perc, prr, qr

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_cp, ac_ct, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, ei, pn, en, pr, perc, prr, qr)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei = min(ac_pet(k), ac_prcp(k))

                    pn = max(0._sp, ac_prcp(k) - ei)

                    en = ac_pet(k) - ei

                    call gr_production(pn, en, ac_cp(k), 1000._sp, ac_hp(k), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = pr + perc

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                ac_qt(k) = qr

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine grd_time_step

    subroutine loieau_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_ca, ac_cc, ac_kb, &
    & ac_ha, ac_hc, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in):: ac_mlt
        real(sp), dimension(mesh%nac), intent(in):: ac_ca, ac_cc, ac_kb
        real(sp), dimension(mesh%nac), intent(inout):: ac_ha, ac_hc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: ei, pn, en, pr, perc, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ca, ac_cc, ac_kb, ac_ha, ac_hc, ac_qt) &
        !$OMP& private(row, col, k, ei, pn, en, pr, perc, prr, prd, qr, qd)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei = min(ac_pet(k), ac_prcp(k))

                    pn = max(0._sp, ac_prcp(k) - ei)

                    en = ac_pet(k) - ei

                    call gr_production(pn, en, ac_ca(k), 9._sp/4._sp, ac_ha(k), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = 0.9_sp*(pr + perc)
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(4._sp, ac_prcp(k), prr, ac_cc(k), ac_hc(k), qr)

                qd = max(0._sp, prd)

                ac_qt(k) = ac_kb(k)*(qr + qd)

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine loieau_time_step

end module md_gr_operator
