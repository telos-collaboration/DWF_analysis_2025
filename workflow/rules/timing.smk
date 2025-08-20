# TODO Test and fix
rule timing_plot:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/time_elapsed_autocorr_plaquette.py",
        data=rules.combined_mobius_csv.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/time_plot_plaquette.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --data {input.data} --plot_styles {input.plot_styles} --output_file {output.plot}"


def Ls_scan_data(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/{filename}.csv".format(
            subdir=datum.name,
            filename=filename,
        )
        for datum in metadata.itertuples()
        for filename in ["spectrum", "timing"]
        if datum.use_in_Ls_scan and datum.beta == float(wildcards.beta)
    ]


rule combined_Ls_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=Ls_scan_data,
    output:
        csv="intermediary_data/Ls_scan_{beta}.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --output_file {output.csv}"


# TODO Test and fix
rule Ls_scan_timing:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/plot_mres_vs_diracapps.py",
        data=rules.combined_Ls_scan.output.csv,
    output:
        plot="assets/plots/Ls_scan_beta{beta}.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --data {input.data} --plot_styles {input.plot_styles} --output_file {output.plot} --title '$\\beta = {wildcards.beta}$"
