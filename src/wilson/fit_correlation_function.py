from argparse import ArgumentParser, FileType
from numpy import pi
import pandas as pd

from .plots import do_eff_mass_plot, do_correlator_plot, set_plot_defaults
from .data import get_target_correlator, get_output_filename, get_plaquettes
from .bootstrap import (
    bootstrap_correlators,
    bootstrap_eff_masses,
    BOOTSTRAP_SAMPLE_COUNT,
    basic_bootstrap,
)
from .fitting import minimize_chisquare, ps_fit_form, ps_av_fit_form, v_fit_form

from ..provenance import get_basic_metadata, text_metadata

channel_set_options = {
    "g5": (("g5",), ("g5_g0g5_re",)),
    "g5_mass": (("g5",),),
    "id": (("id",),),
    "gk": (("g1", "g2", "g3"),),
    "g5gk": (("g5g1", "g5g2", "g5g3"),),
    "g0gk": (("g0g1", "g0g2", "g0g3"),),
    "g0g5gk": (("g0g5g1", "g0g5g2", "g0g5g3"),),
}
correlator_names_options = {
    "g5": ("g5", "g5_g0g5_re"),
    "g5_mass": ("g5",),
    "id": ("id",),
    "gk": ("gk"),
    "g5gk": ("g5gk"),
    "g0gk": ("g0gk"),
    "g0g5gk": ("g0g5gk"),
}
channel_latexes_options = {
    "g5": (r"\gamma_5,\gamma_5", r"\gamma_0\gamma_5,\gamma_5"),
    "g5_mass": (r"\gamma_5,\gamma_5",),
    "id": (r"1,1",),
    "gk": (r"\gamma_k,\gamma_k",),
    "g5gk": (r"\gamma_5 \gamma_k,\gamma_5 \gamma_k",),
    "g0gk": (r"\gamma_0 \gamma_k,\gamma_0 \gamma_k",),
    "g0g5gk": (r"\gamma_0 \gamma_5 \gamma_k,\gamma_0 \gamma_5 \gamma_k",),
}
fit_forms_options = {
    "g5": (ps_fit_form, ps_av_fit_form),
    "g5_mass": (v_fit_form,),
    "id": (v_fit_form,),
    "gk": (v_fit_form,),
    "g5gk": (v_fit_form,),
    "g0gk": (v_fit_form,),
    "g0g5gk": (v_fit_form,),
}
symmetries_options = {
    "g5": (+1, -1),
    "g5_mass": (+1,),
    "id": (+1,),
    "gk": (+1,),
    "g5gk": (+1,),
    "g0gk": (+1,),
    "g0g5gk": (+1,),
}
parameter_range_options = {
    "g5": ((0.01, 5), (0, 5), (0, 5)),
    "g5_mass": ((0.01, 5), (0, 5)),
    "id": ((0.01, 5), (0, 5)),
    "gk": ((0.01, 5), (0, 5)),
    "g5gk": ((0.01, 5), (0, 5)),
    "g0gk": ((0.01, 5), (0, 5)),
    "g0g5gk": ((0.01, 5), (0, 5)),
}
quantity_options = {
    "g5": ("mass", "decay_const", "amplitude", "chisquare"),
    "g5_mass": ("mass", "decay_const", "chisquare"),
    "id": ("mass", "decay_const", "chisquare"),
    "gk": ("mass", "decay_const", "chisquare"),
    "g5gk": ("mass", "decay_const", "chisquare"),
    "g0gk": ("mass", "decay_const", "chisquare"),
    "g0g5gk": ("mass", "decay_const", "chisquare"),
}


class Incomplete(Exception):
    pass


def process_correlator(
    correlator_filename,
    channel_name,
    channel_set,
    channel_latexes,
    symmetries,
    correlator_names,
    fit_forms,
    NT,
    NS,
    parameter_ranges,
    ensemble_selection=0,
    initial_configuration=0,
    configuration_separation=1,
    bootstrap_sample_count=BOOTSTRAP_SAMPLE_COUNT,
    plateau_start=None,
    plateau_end=None,
    eff_mass_plot_ymin=None,
    eff_mass_plot_ymax=None,
    correlator_lowerbound=None,
    correlator_upperbound=None,
    optimizer_intensity="default",
    raw_correlators=True,
    _iter=0,
    maxiter=4,
    output_effmass_plot_with_fit=None,
    output_effmass_plot=None,
    output_correlator_plot=None,
    output_centrally_fitted_correlator_plot=None,
):
    set_plot_defaults()
    target_correlator_sets, valence_masses = get_target_correlator(
        correlator_filename,
        channel_set,
        NT,
        NS,
        symmetries,
        ensemble_selection,
        initial_configuration,
        configuration_separation,
        from_raw=raw_correlators,
    )

    fit_results_set = []

    for target_correlators, valence_mass in zip(target_correlator_sets, valence_masses):
        (
            bootstrap_mean_correlators,
            bootstrap_error_correlators,
            bootstrap_correlator_samples_set,
        ) = bootstrap_correlators(target_correlators)

        bootstrap_mean_eff_masses, bootstrap_error_eff_masses = bootstrap_eff_masses(
            bootstrap_correlator_samples_set
        )

        do_eff_mass_plot(
            bootstrap_mean_eff_masses[0],
            bootstrap_error_eff_masses[0],
            output_effmass_plot,
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax,
        )

        for (
            correlator_name,
            channel_latex,
            bootstrap_mean_correlator,
            bootstrap_error_correlator,
        ) in zip(
            correlator_names,
            channel_latexes,
            bootstrap_mean_correlators,
            bootstrap_error_correlators,
        ):
            do_correlator_plot(
                bootstrap_mean_correlator,
                bootstrap_error_correlator,
                output_correlator_plot,
                channel_latex,
            )

        if not (plateau_start and plateau_end):
            continue

        (fit_results, (chisquare_value, chisquare_error), _) = minimize_chisquare(
            bootstrap_correlator_samples_set,
            bootstrap_mean_correlators,
            fit_forms,
            parameter_ranges,
            plateau_start,
            plateau_end,
            NT,
            fit_means=True,
            intensity=optimizer_intensity,
        )
        fit_result_values = tuple(fit_result[0] for fit_result in fit_results)

        for (
            correlator_name,
            channel_latex,
            fit_form,
            bootstrap_mean_correlator,
            bootstrap_error_correlator,
        ) in zip(
            correlator_names,
            channel_latexes,
            fit_forms,
            bootstrap_mean_correlators,
            bootstrap_error_correlators,
        ):
            do_correlator_plot(
                bootstrap_mean_correlator,
                bootstrap_error_correlator,
                output_centrally_fitted_correlator_plot,
                channel_latex,
                fit_function=fit_form,
                fit_params=(*fit_result_values, NT),
                fit_legend="Fit of central values",
                t_lowerbound=plateau_start - 3.5,
                t_upperbound=plateau_end - 0.5,
                corr_upperbound=correlator_upperbound,
                corr_lowerbound=correlator_lowerbound,
            )

        (fit_results, (chisquare_value, chisquare_error), final_chisquare) = (
            minimize_chisquare(
                bootstrap_correlator_samples_set,
                bootstrap_mean_correlators,
                fit_forms,
                parameter_ranges,
                plateau_start,
                plateau_end,
                NT,
                fit_means=False,
            )
        )
        (mass, mass_error), *_ = fit_results
        fit_results_set.append((fit_results, final_chisquare))

        do_eff_mass_plot(
            bootstrap_mean_eff_masses[0],
            bootstrap_error_eff_masses[0],
            output_effmass_plot_with_fit,
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax,
            m=mass,
            m_error=mass_error,
            tmin=plateau_start - 0.5,
            tmax=plateau_end - 0.5,
        )

    if not (plateau_start and plateau_end):
        raise Incomplete(
            "Effective mass plot has been generated. "
            "Now specify the start and end of the plateau to "
            "perform the fit."
        )
    if chisquare_error > chisquare_value:
        if optimizer_intensity == "default":
            optimizer_intensity = "intense_de"
            print("    Trying intense_de")
        else:
            if _iter >= maxiter:
                print(
                    "    WARNING: max iters exceeded with "
                    f"{bootstrap_sample_count} samples."
                )
                return fit_results_set, valence_masses
            else:
                print(
                    "    Increasing boostrap samples to " f"{bootstrap_sample_count}..."
                )
                _iter = _iter + 1
        return process_correlator(
            correlator_filename,
            channel_name,
            channel_set,
            channel_latexes,
            symmetries,
            correlator_names,
            fit_forms,
            NT,
            NS,
            parameter_ranges,
            ensemble_selection=ensemble_selection,
            initial_configuration=initial_configuration,
            configuration_separation=configuration_separation,
            bootstrap_sample_count=bootstrap_sample_count * 2,
            plateau_start=plateau_start,
            plateau_end=plateau_end,
            eff_mass_plot_ymin=eff_mass_plot_ymin,
            eff_mass_plot_ymax=eff_mass_plot_ymax,
            correlator_lowerbound=correlator_lowerbound,
            correlator_upperbound=correlator_upperbound,
            optimizer_intensity=optimizer_intensity,
            raw_correlators=raw_correlators,
            output_effmass_plot_with_fit=output_effmass_plot_with_fit,
            output_effmass_plot=output_effmass_plot,
            output_correlator_plot=output_correlator_plot,
            output_centrally_fitted_correlator_plot=output_centrally_fitted_correlator_plot,
            _iter=_iter,
            maxiter=maxiter,
        )

    return fit_results_set, valence_masses


def plot_measure_and_save_mesons(
    simulation_descriptor,
    correlator_filename,
    channel_name,
    meson_parameters=None,
    parameter_date=None,
    output_effmass_plot_with_fit=None,
    output_effmass_plot=None,
    output_correlator_plot=None,
    output_centrally_fitted_correlator_plot=None,
):
    # Distinguish between g5 with and without decay constant in analysis
    # But make them use same name in database to allow easy tabulation
    # and plotting
    db_channel_name = channel_name

    if not meson_parameters:
        meson_parameters = {}

    if meson_parameters.pop("no_decay_const", False) and channel_name == "g5":
        channel_name = "g5_mass"

    channel_set = channel_set_options[channel_name]
    correlator_names = correlator_names_options[channel_name]
    channel_latexes = channel_latexes_options[channel_name]
    fit_forms = fit_forms_options[channel_name]
    symmetries = symmetries_options[channel_name]
    parameter_ranges = parameter_range_options[channel_name]

    fit_results_set, valence_masses = process_correlator(
        correlator_filename,
        channel_name,
        channel_set,
        channel_latexes,
        symmetries,
        correlator_names,
        fit_forms,
        simulation_descriptor["T"],
        simulation_descriptor["L"],
        initial_configuration=simulation_descriptor.get("initial_configuration", 0),
        output_effmass_plot_with_fit=output_effmass_plot_with_fit,
        output_effmass_plot=output_effmass_plot,
        output_correlator_plot=output_correlator_plot,
        output_centrally_fitted_correlator_plot=output_centrally_fitted_correlator_plot,
        parameter_ranges=parameter_ranges,
        **meson_parameters,
    )

    return valence_masses, fit_results_set


def get_renorm_coefft(
    filename, ensemble_selection, initial_configuration, configuration_separation, beta
):
    # Z_V = 1 + C_F \Delta \tilde{g}^2 / {16 \pi^2}
    # \tilde{g} \langle P \rangle = 8 / \beta
    # \Delta = \Delta_{\Sigma_1} + \Delta_{\gamma_mu}
    delta_sigma_one = -12.82
    delta_gmu = -7.75
    fundamental_casimir = 5 / 4
    plaquettes = get_plaquettes(
        filename, ensemble_selection, initial_configuration, configuration_separation
    )
    coefft_values = 1 + fundamental_casimir * (delta_sigma_one + delta_gmu) * (
        8 / beta / plaquettes
    ) / (16 * pi**2)
    return basic_bootstrap(coefft_values)


def write_output(args, valence_masses, fit_results_set):
    if len(fit_results_set) > 1:
        raise NotImplementedError("Dataset contains multiple valence masses.")

    mass, mass_error = fit_results_set[0][0][0]
    decay_const, decay_const_error = fit_results_set[0][0][1]
    chisquare = fit_results_set[0][1]

    result = {
        "name": args.tag,
        "NT": args.NT,
        "NX": args.NS,
        "NY": args.NS,
        "NZ": args.NS,
        "beta": args.beta,
        "mF": args.mF,
        "configuration_separation": args.configuration_separation,
        "initial_configuration": args.initial_configuration,
        "valence_mass": valence_masses[0],
        f"{args.channel}_mass": mass,
        f"{args.channel}_mass_error": mass_error,
        f"{args.channel}_bare_decay_const": decay_const,
        f"{args.channel}_bare_decay_const_error": decay_const_error,
        f"{args.channel}_chisquare": chisquare,
    }
    if args.channel == "g5":
        renorm_coefft, renorm_coefft_error = get_renorm_coefft(
            args.correlator_filename,
            args.ensemble_selection,
            args.initial_configuration,
            args.configuration_separation,
            args.beta,
        )
        result["g5_decay_const"] = renorm_coefft * decay_const
        result["g5_decay_const_error"] = (
            result["g5_decay_const"]
            * (
                (decay_const_error / decay_const) ** 2
                + (renorm_coefft_error / renorm_coefft) ** 2
            )
            ** 0.5
        )

    if len(fit_results_set[0][0]) > 2:
        amplitude, amplitude_error = fit_results_set[0][0][2]
        result[f"{args.channel}_amplitude"] = amplitude
        result[f"{args.channel}_amplitude_error"] = amplitude_error

    df = pd.DataFrame([result])
    print(text_metadata(get_basic_metadata(), comment_char="#"), file=args.output_file)
    print(df.to_csv(index=False), file=args.output_file)


def main():
    parser = ArgumentParser()

    parser.add_argument("--correlator_filename", required=True)
    parser.add_argument("--channel", choices=("g5", "gk", "g5gk"), required=True)
    parser.add_argument("--NT", required=True, type=int)
    parser.add_argument("--NS", required=True, type=int)
    parser.add_argument("--beta", required=False, type=float, default=None)
    parser.add_argument("--mF", required=False, type=float, default=None)
    parser.add_argument("--configuration_separation", default=1, type=int)
    parser.add_argument("--delta_traj", default=1, type=int)
    parser.add_argument("--initial_configuration", default=0, type=int)
    # ensemble_selection can range from 0 to configuration_separation
    parser.add_argument("--ensemble_selection", default=0, type=int)
    parser.add_argument("--bootstrap_sample_count", default=200, type=int)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--eff_mass_plot_ymin", default=None, type=float)
    parser.add_argument("--eff_mass_plot_ymax", default=None, type=float)
    parser.add_argument("--plateau_start", default=None, type=int)
    parser.add_argument("--plateau_end", default=None, type=int)
    parser.add_argument("--correlator_lowerbound", default=0.0, type=float)
    parser.add_argument("--correlator_upperbound", default=None, type=float)
    parser.add_argument(
        "--optimizer_intensity", default="default", choices=("default", "intense")
    )
    parser.add_argument("--ignore", action="store_true")
    parser.add_argument("--no_decay_const", action="store_true")
    parser.add_argument("--raw_correlators", action="store_true")
    parser.add_argument("--output_file", type=FileType("w"), default="-")
    parser.add_argument("--output_effmass_plot_with_fit", default=None)
    parser.add_argument("--output_effmass_plot", default=None)
    parser.add_argument("--output_correlator_plot", default=None)
    parser.add_argument("--output_centrally_fitted_correlator_plot", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    meson_parameters = {
        key: args.__dict__[key]
        for key in [
            "ensemble_selection",
            "eff_mass_plot_ymin",
            "eff_mass_plot_ymax",
            "plateau_start",
            "plateau_end",
            "correlator_lowerbound",
            "correlator_upperbound",
            "optimizer_intensity",
            "no_decay_const",
            "raw_correlators",
            "configuration_separation",
        ]
    }
    simulation_descriptor = {
        "L": args.NS,
        "T": args.NT,
        "delta_traj": args.delta_traj,
        "initial_configuration": args.initial_configuration,
    }
    if not args.ignore:
        try:
            valence_masses, fit_results_set = plot_measure_and_save_mesons(
                simulation_descriptor,
                args.correlator_filename,
                args.channel,
                meson_parameters=meson_parameters,
                output_effmass_plot_with_fit=args.output_effmass_plot_with_fit,
                output_effmass_plot=args.output_effmass_plot,
                output_correlator_plot=args.output_correlator_plot,
                output_centrally_fitted_correlator_plot=args.output_centrally_fitted_correlator_plot,
            )
        except Incomplete as ex:
            print("ANALYSIS NOT YET COMPLETE")
            print(ex)

        else:
            write_output(args, valence_masses, fit_results_set)
            if not args.silent:
                for valence_mass, fit_results in zip(valence_masses, fit_results_set):
                    mass, mass_error = fit_results[0][0]
                    decay_const, decay_const_error = fit_results[0][1]
                    if len(fit_results[0]) > 2:
                        amplitude, amplitude_error = fit_results[0][2]
                    chisquare_value = fit_results[1]

                    print(f"{args.channel} mass: {mass} ± {mass_error}")
                    print(
                        f"{args.channel} decay constant: "
                        f"{decay_const} ± {decay_const_error}"
                    )
                    if len(fit_results[0]) > 2:
                        print(
                            f"{args.channel} amplitude: "
                            f"{amplitude} ± {amplitude_error}"
                        )
                    print(f"{args.channel} chi-square: " f"{chisquare_value}")


if __name__ == "__main__":
    main()
