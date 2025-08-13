import yaml

with open(config["wilson_metadata"], "r") as f:
    wilson_metadata = yaml.safe_load(f)

wilson_subdir_template = "{NT}x{NX}x{NY}x{NZ}b{beta}m{mF}"
wilson_outdir_template = f"intermediary_data/wilson/{wilson_subdir_template}"


def get_metadata(wildcards, key):
    matches = [
        (name, ensemble)
        for name, ensemble in wilson_metadata.items()
        if "T" in ensemble
        and ensemble["T"] == int(wildcards["NT"])
        and ensemble["L"] == int(wildcards["NX"])
        and ensemble["beta"] == float(wildcards["beta"])
        and ensemble["m"] == float(wildcards["mF"])
    ]
    ((name, ensemble),) = matches

    if key in ["initial_configuration", "delta_traj"]:
        return ensemble[key]
    if key == "tag":
        return name
    return ensemble["measure_mesons"][wildcards["channel"]][key]


rule wilson_mass:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
        configuration_separation=lambda wildcards: get_metadata(
            wildcards, "configuration_separation"
        ),
        ensemble_selection=lambda wildcards: get_metadata(
            wildcards, "ensemble_selection"
        ),
        initial_configuration=lambda wildcards: get_metadata(
            wildcards, "initial_configuration"
        ),
        delta_traj=lambda wildcards: get_metadata(wildcards, "delta_traj"),
        plateau_start=lambda wildcards: get_metadata(wildcards, "plateau_start"),
        plateau_end=lambda wildcards: get_metadata(wildcards, "plateau_end"),
        tag=lambda wildcards: get_metadata(wildcards, "tag"),
    input:
        script="src/wilson/fit_correlation_function.py",
        correlators=f"{config['wilson_data_dir']}/{wilson_subdir_template}/out_corr",
    output:
        data=f"{wilson_outdir_template}/{{channel}}_mass.csv",
        effmass_plot=f"{wilson_outdir_template}/{{channel}}_effmass.pdf",
        effmass_plot_with_fit=f"{wilson_outdir_template}/{{channel}}_effmass_withfit.pdf",
        correlator_plot=f"{wilson_outdir_template}/{{channel}}_correlator.pdf",
        fitted_correlator_plot=f"{wilson_outdir_template}/{{channel}}_centrally_fitted_correlator.pdf",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --correlator_filename {input.correlators} --channel {wildcards.channel} --NT {wildcards.NT} --NS {wildcards.NX} --beta {wildcards.beta} --mF {wildcards.mF} --delta_traj {params.delta_traj} --configuration_separation {params.configuration_separation} --initial_configuration {params.initial_configuration} --ensemble_selection {params.ensemble_selection} --silent --plateau_start {params.plateau_start} --plateau_end {params.plateau_end} --raw_correlators --output_file {output.data} --output_effmass_plot_with_fit {output.effmass_plot_with_fit} --output_effmass_plot {output.effmass_plot} --output_correlator_plot {output.correlator_plot} --output_centrally_fitted_correlator_plot {output.fitted_correlator_plot} --tag {params.tag}"


rule wilson_WF_and_Q:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
        tag=lambda wildcards: get_metadata(wildcards, "tag"),
    input:
        script="src/WF_and_Q.py",
        data=f"{config['wilson_data_dir']}/{wilson_subdir_template}/out_wflow",
    output:
        summary=f"{wilson_outdir_template}/WF_summary.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --tag {params.tag} --output_file {output.summary} --W0 {config[W0]} --fileformat hirep"


def get_wilson_outputs(wildcards, key, filename):
    ensembles = [
        ensemble
        for name, ensemble in wilson_metadata.items()
        if ensemble.get(f"measure_{key}")
    ]
    return [
        f"{wilson_outdir_template}/{filename}.csv".format(
            NT=ensemble["T"],
            NX=ensemble["L"],
            NY=ensemble["L"],
            NZ=ensemble["L"],
            beta=ensemble["beta"],
            mF=ensemble["m"],
        )
        for ensemble in ensembles
    ]


rule combine_wilson_csvs:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        flow_csvs=lambda wildcards: get_wilson_outputs(wildcards, "gflow", "WF_summary"),
        spectrum_csvs=lambda wildcards: get_wilson_outputs(
            wildcards, "mesons", "g5_mass"
        ),
    output:
        csv="data_assets/wilson_summary.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.flow_csvs} {input.spectrum_csvs} --output_file {output.csv}"
