import numpy as np


def fold_correlators(correlators):
    num_time_slices = correlators.shape[0]
    num_configs = correlators.shape[1]
    # print('num_configs: ', num_configs)
    # print('num_time_slices: ',num_time_slices)
    num_half_slices = num_time_slices // 2 + 1

    folded_correlators = np.zeros((num_half_slices, num_configs))

    folded_correlators[0] = correlators[0]
    for i in range(1, num_half_slices):
        if correlators[i][1] * correlators[num_time_slices - i][1] < 0:
            folded_correlators[i] = 0.5 * (
                -correlators[i] + correlators[num_time_slices - i]
            )
        else:
            folded_correlators[i] = 0.5 * (
                correlators[i] + correlators[num_time_slices - i]
            )

    return folded_correlators


def fold_correlators_ZA(correlators):
    num_time_slices = correlators.shape[0]
    num_configs = correlators.shape[1]
    # print('num_configs: ', num_configs)
    # print('num_time_slices: ',num_time_slices)
    num_half_slices = num_time_slices // 2
    print(num_half_slices)
    folded_correlators = np.zeros((num_half_slices, num_configs))

    for i in range(0, num_half_slices):
        # print('correlators[i]: ', correlators[i])
        # print('correlators[num_time_slices - 1 - i]: ', correlators[num_time_slices - 1 - i])
        folded_correlators[i] = 0.5 * (
            -correlators[i] + correlators[num_time_slices - 1 - i]
        )

    return folded_correlators


def bootstrap_effective_mass(C1, C2, num_bootstrap=1000):
    Nt, Nconf = C1.shape
    bootstrap_masses = np.zeros((num_bootstrap, Nt))

    for i in range(num_bootstrap):
        bootstrap_indices = np.random.choice(Nconf, size=Nconf, replace=True)
        bootstrap_C1 = C1[:, bootstrap_indices]
        bootstrap_C2 = C2[:, bootstrap_indices]
        bootstrap_mass_sample = np.mean(bootstrap_C1 / bootstrap_C2, axis=1)
        bootstrap_masses[i] = bootstrap_mass_sample

    mean_mass = np.mean(bootstrap_masses, axis=0)
    error_mass = np.std(bootstrap_masses, axis=0)
    # print(error_mass)
    return mean_mass, error_mass


def bootstrap_effective_mass2(C1, C2, num_bootstrap=2000):
    Nt, Nconf = C1.shape
    bootstrap_masses = np.zeros((num_bootstrap, Nt))

    for i in range(num_bootstrap):
        bootstrap_indices = np.random.choice(Nconf, size=Nconf, replace=True)
        bootstrap_C1 = C1[:, bootstrap_indices]
        bootstrap_C2 = C2[:, bootstrap_indices]
        bootstrap_mass_sample = np.mean(bootstrap_C2, axis=1)
        bootstrap_masses[i] = bootstrap_mass_sample

    mean_mass = np.mean(bootstrap_masses, axis=0)
    error_mass = np.std(bootstrap_masses, axis=0)
    # print(error_mass)
    return mean_mass, error_mass


def jackknife_effective_mass_block(C1, C2, block_size_fraction=0.2):
    Nt, Nconf = C1.shape
    block_size = 1
    # block_size = int(Nconf * block_size_fraction)
    num_blocks = Nconf // block_size
    jackknife_masses = np.zeros((num_blocks, Nt))

    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        indices_to_keep = np.setdiff1d(
            np.arange(Nconf), np.arange(start_index, end_index)
        )
        jackknife_C1 = np.mean(C1[:, indices_to_keep], axis=1)
        jackknife_C2 = np.mean(C2[:, indices_to_keep], axis=1)
        jackknife_mass_sample = jackknife_C1 / jackknife_C2
        jackknife_masses[i] = jackknife_mass_sample

    mean_mass = np.mean(jackknife_masses, axis=0)
    # pseudo_values = num_blocks * mean_mass - (num_blocks - 1) * jackknife_masses
    # error_mass = np.sqrt((num_blocks - 1) * np.var(pseudo_values, axis=0))
    error_mass = np.sqrt((num_blocks - 1) * np.var(jackknife_masses, axis=0))

    return mean_mass, error_mass


def jackknife_effective_mass_block2(C1, C2, block_size_fraction=0.1):
    Nt, Nconf = C1.shape
    block_size = 1
    # block_size = int(Nconf * block_size_fraction)
    num_blocks = Nconf // block_size
    jackknife_masses = np.zeros((num_blocks, Nt))

    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        indices_to_keep = np.setdiff1d(
            np.arange(Nconf), np.arange(start_index, end_index)
        )
        jackknife_C1 = C1[:, indices_to_keep]
        jackknife_C2 = C2[:, indices_to_keep]
        # print(jackknife_C2.shape)
        jackknife_mass_sample = np.mean(jackknife_C2, axis=1)
        jackknife_masses[i] = jackknife_mass_sample

    mean_mass = np.mean(jackknife_masses, axis=0)
    # pseudo_values = num_blocks * mean_mass - (num_blocks - 1) * jackknife_masses
    # error_mass = np.sqrt((num_blocks - 1) * np.var(pseudo_values, axis=0))
    error_mass = np.sqrt((num_blocks - 1) * np.var(jackknife_masses, axis=0))

    return mean_mass, error_mass


def jackknife_effective_mass_block3(C1, block_size_fraction=0.1):
    Nt, Nconf = C1.shape
    block_size = 1
    # block_size = int(Nconf * block_size_fraction)
    num_blocks = Nconf // block_size
    jackknife_masses = np.zeros((num_blocks, Nt - 1))
    tmp_array = np.empty((C1.shape[0] - 1))
    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        indices_to_keep = np.setdiff1d(
            np.arange(Nconf), np.arange(start_index, end_index)
        )
        jackknife_C1 = np.mean(C1[:, indices_to_keep], axis=1)

        for t in range(0, C1.shape[0] - 1):
            # tmp_array[t] = np.arcosh(( jackknife_C1[t+1] + jackknife_C1[t-1] / (2*jackknife_C1[t])))
            tmp_array[t] = np.log((jackknife_C1[t] / (jackknife_C1[t + 1])))
        # print(tmp_array)
        jackknife_masses[i] = tmp_array

    mean_mass = np.mean(jackknife_masses, axis=0)
    # pseudo_values = num_blocks * mean_mass - (num_blocks - 1) * jackknife_masses
    # error_mass = np.sqrt((num_blocks - 1) * np.var(pseudo_values, axis=0))
    error_mass = np.sqrt((num_blocks - 1) * np.var(jackknife_masses, axis=0))

    return mean_mass, error_mass, jackknife_masses


def jackknife_effective_mass_block4(C1, NT, block_size_fraction=0.1):
    Nt, Nconf = C1.shape
    block_size = 1
    # block_size = int(Nconf * block_size_fraction)
    num_blocks = Nconf // block_size
    jackknife_masses = np.zeros((num_blocks, Nt - 1))
    tmp_array = np.empty((C1.shape[0] - 1))

    from scipy.optimize import fsolve

    def equation(meff, C_nt, C_nt_plus_1, nt, NT):
        A = nt - NT / 2
        B = nt + 1 - NT / 2
        lhs = C_nt / C_nt_plus_1
        rhs = np.cosh(meff * A) / np.cosh(meff * B)
        # rhs = np.sinh(meff * A) / np.sinh(meff * B)
        return lhs - rhs

    def find_meff(C_nt, C_nt_plus_1, nt, NT):
        # Initial guess for meff
        initial_guess = 0.5
        (meff_solution,) = fsolve(
            equation, initial_guess, args=(C_nt, C_nt_plus_1, nt, NT)
        )
        return meff_solution

    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        indices_to_keep = np.setdiff1d(
            np.arange(Nconf), np.arange(start_index, end_index)
        )
        # print(indices_to_keep)
        jackknife_C1 = np.mean(C1[:, indices_to_keep], axis=1)

        for t in range(0, C1.shape[0] - 1):
            # tmp_array[t] = np.arcosh(( jackknife_C1[t+1] + jackknife_C1[t-1] / (2*jackknife_C1[t])))
            tmp_array[t] = find_meff(jackknife_C1[t], jackknife_C1[t + 1], t, NT)
            # tmp_array[t] = np.log((jackknife_C1[t] / (jackknife_C1[t+1])))
        # print(tmp_array)
        jackknife_masses[i] = tmp_array

    mean_mass = np.mean(jackknife_masses, axis=0)
    # pseudo_values = num_blocks * mean_mass - (num_blocks - 1) * jackknife_masses
    # error_mass = np.sqrt((num_blocks - 1) * np.var(pseudo_values, axis=0))
    error_mass = np.sqrt((num_blocks - 1) * np.var(jackknife_masses, axis=0))
    return mean_mass, error_mass, jackknife_masses


def jackknife_effective_mass_block_ZA(C1, C2, block_size_fraction=0.1):
    Nt, Nconf = C1.shape
    block_size = 1
    # block_size = int(Nconf * block_size_fraction)
    num_blocks = Nconf // block_size
    jackknife_masses = np.zeros((num_blocks, Nt - 1))
    tmp_array = np.empty((C1.shape[0] - 1))
    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        indices_to_keep = np.setdiff1d(
            np.arange(Nconf), np.arange(start_index, end_index)
        )
        jackknife_C1 = C1[:, indices_to_keep]
        jackknife_C2 = C2[:, indices_to_keep]
        jackknife_mass_sample1 = np.mean(jackknife_C1, axis=1)
        jackknife_mass_sample2 = np.mean(jackknife_C2, axis=1)
        # jackknife_mass_sample2[0] = - jackknife_mass_sample2[0]
        # print(jackknife_mass_sample1)
        # print(jackknife_mass_sample2)
        for t in range(0, C1.shape[0] - 1):
            tmp_array[t] = 0.5 * (
                (jackknife_mass_sample1[t + 1] + jackknife_mass_sample1[t])
                / (2 * jackknife_mass_sample2[t + 1])
                + 2
                * jackknife_mass_sample1[t + 1]
                / (jackknife_mass_sample2[t + 1] + jackknife_mass_sample2[t + 2])
            )
        jackknife_masses[i] = tmp_array

    mean_mass = np.mean(jackknife_masses, axis=0)
    # pseudo_values = num_blocks * mean_mass - (num_blocks - 1) * jackknife_masses
    # error_mass = np.sqrt((num_blocks - 1) * np.var(pseudo_values, axis=0))
    error_mass = np.sqrt((num_blocks - 1) * np.var(jackknife_masses, axis=0))

    return mean_mass, error_mass, jackknife_masses


def bootstrap_effective_mass3(C1, num_bootstrap=1000):
    Nt, Nconf = C1.shape
    bootstrap_masses = np.zeros((num_bootstrap, Nt))

    for i in range(num_bootstrap):
        bootstrap_indices = np.random.choice(Nconf, size=Nconf, replace=True)
        bootstrap_C1 = C1[:, bootstrap_indices]
        bootstrap_mass_sample = np.mean(bootstrap_C1, axis=1)
        bootstrap_masses[i] = bootstrap_mass_sample

    mean_mass = np.mean(bootstrap_masses, axis=0)
    error_mass = np.std(bootstrap_masses, axis=0)
    # print(error_mass)
    return mean_mass, error_mass


from scipy.optimize import minimize


def correlated_chi_square(params, measurements, covariance_matrix):
    avg = params[0]
    chi_sq = 0
    for i in range(len(measurements)):
        for j in range(len(measurements)):
            chi_sq += (
                (measurements[i] - avg)
                * covariance_matrix[i, j]
                * (measurements[j] - avg)
            )
    return chi_sq


def perform_correlated_fit(
    effective_mass, covariance_matrix, ti, tf, num_bootstrap=100
):
    # Select the range of effective masses for the fit
    indices = np.arange(ti, tf + 1)
    # print(indices)
    selected_masses = effective_mass[indices]
    selected_covariance = covariance_matrix[np.ix_(indices, indices)]

    # Perform fits on multiple bootstrap samples
    fitted_avgs = []
    chi_squares = []
    for _ in range(num_bootstrap):
        # Generate bootstrap sample
        bootstrap_indices = np.random.choice(
            len(selected_masses), size=len(selected_masses), replace=True
        )
        bootstrap_masses = selected_masses[bootstrap_indices]

        # Fit the bootstrap sample
        initial_guess = [np.mean(bootstrap_masses)]
        min_func = lambda params: correlated_chi_square(
            params, bootstrap_masses, selected_covariance
        )
        result = minimize(min_func, initial_guess, method="Nelder-Mead")
        fitted_avg = result.x[0]
        fitted_avgs.append(fitted_avg)

        # Compute correlated chi square for the fit
        chi_square = min_func(result.x)
        chi_squares.append(chi_square)

    # Calculate the error as the standard deviation of the fitted averages
    error_avg = np.std(fitted_avgs)

    # Calculate the final fitted average
    fitted_avg = np.mean(fitted_avgs)

    # Calculate the average correlated chi square
    avg_chi_square = np.mean(chi_squares)

    return fitted_avg, error_avg, avg_chi_square


def effective_mass(corr, NT):
    mean, err, tmp_array = jackknife_effective_mass_block4(corr, NT)
    # mean, err = bootstrap_effective_mass3(corr)
    return mean, err, tmp_array
