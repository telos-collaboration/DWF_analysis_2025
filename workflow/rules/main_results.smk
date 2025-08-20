def WF_csvs(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/WF_summary.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.gflow_step > 0
    ]


rule combine_WF_Q:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=WF_csvs,
    output:
        csv="data_assets/WF_measurements.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --output_file {output.csv}"


def spectrum_csvs(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/spectrum.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.have_spectrum
    ]


rule combine_spectrum:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=spectrum_csvs,
    output:
        csv="data_assets/plateau_fits_results.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --output_file {output.csv}"


def hmc_csvs(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/timing.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.have_hmc
    ]


rule combine_hmc:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=hmc_csvs,
    output:
        csv="data_assets/hmc_timing_results.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --output_file {output.csv}"


rule combined_mobius_csv:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        hmc=rules.combine_hmc.output.csv,
        wflow=rules.combine_WF_Q.output.csv,
        spectrum=rules.combine_spectrum.output.csv,
    output:
        csv="data_assets/mobius_results.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.hmc} {input.wflow} {input.spectrum} --output_file {output.csv}"


rule GMOR_plots:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/GMOR_and_mV_mPS_ratio.py",
        data=rules.combined_mobius_csv.output.csv,
        plot_styles=config["plot_styles"],
    output:
        GMOR_mPS="assets/plots/GMOR_w0m0_vs_w0m_PS_{beta}.{filetype}",
        GMOR_fpi="assets/plots/GMOR_w0m0_vs_w0fpi_{beta}.{filetype}",
        GMOR_mPSfpi="assets/plots/GMOR_w0m0_vs_w0_m_PS_fpi_{beta}.{filetype}",
        bare="assets/plots/m0_vs_m_V_m_PS_{beta}.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --plot_styles {input.plot_styles} --data {input.data} --beta {wildcards.beta} --output_file_GMOR_mPS {output.GMOR_mPS} --output_file_GMOR_fpi {output.GMOR_fpi} --output_file_GMOR_mPSfpi {output.GMOR_mPSfpi} --output_file_bare {output.bare}"


rule NLO_w0_single_beta:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/NLO_w0_single.py",
        data=rules.combined_mobius_csv.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/NLO_mPS_w0_beta_{beta}.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --plot_styles {input.plot_styles} --data {input.data} --beta {wildcards.beta} --output_file {output.plot}"


rule NLO_w0_summary:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/NLO_w0.py",
        data=rules.combined_mobius_csv.output.csv,
        plot_styles=config["plot_styles"],
        wilson_csv=rules.combine_wilson_csvs.output.csv,
    output:
        plot="assets/plots/chiral_aoverw0_vs_mPS.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --plot_styles {input.plot_styles} --data {input.data} --wilson_results {input.wilson_csv} --output_file {output.plot}"


rule chiPT:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/perform_chiPT.py",
        data=rules.combined_mobius_csv.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/fit_w0_mPS_mV.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --plot_styles {input.plot_styles} --data {input.data} --output_file {output.plot}"
