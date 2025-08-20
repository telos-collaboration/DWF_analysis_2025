def mres_scan_data(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/spectrum.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.use_in_parameter_tuning
        and (
            datum.mF == float(wildcards.mF)
            # Add mass scan plot
            or (
                wildcards.mF == "0.06"
                and datum.Ls == 8
                and datum.beta == 6.8
                and datum.a5 == 1.0
                and datum.alpha == 2.0
                and datum.M5 == 1.8
            )
        )
    ]


rule collate_mres_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=mres_scan_data,
        metadata=config["metadata"],
    output:
        csv="intermediary_data/mres_scan_mF{mF}.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


rule mres_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/plot_mres_scans.py",
        data=rules.collate_mres_scan.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/mres_scan_mf{mF}.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --data {input.data} --plot_styles {input.plot_styles} --output_file {output.plot}"


def Ls_scan_data(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/spectrum.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.use_in_finite_volume_study
        and datum.mF == float(wildcards.mF)
        and datum.beta == float(wildcards.beta)
        and datum.Nt == 32
        and datum.Nx == 24
    ]


rule collate_Ls_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=Ls_scan_data,
        metadata=config["metadata"],
    output:
        csv="intermediary_data/Ls_scan_beta{beta}_mF{mF}.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


rule PS_Ls_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/plot_Ls_scan.py",
        data=rules.collate_Ls_scan.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/Ls_scan_beta{beta}_mF{mF}.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --data {input.data} --plot_styles {input.plot_styles} --output_file {output.plot}"


def volume_scan_data(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/spectrum.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.use_in_finite_volume_study
        and datum.mF == float(wildcards.mF)
        and datum.beta == float(wildcards.beta)
        and datum.Ls == 8
    ]


rule collate_volume_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=volume_scan_data,
        metadata=config["metadata"],
    output:
        csv="intermediary_data/volume_scan_beta{beta}_mF{mF}.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


rule finite_volume_scan:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/plot_L_scan.py",
        data=rules.collate_volume_scan.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/finitevolume_scan_beta{beta}_mF{mF}.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --data {input.data} --plot_styles {input.plot_styles} --output_file {output.plot}"


def mres_scan_largeLs_data(wildcards):
    return [
        "intermediary_data/mdwf/{subdir}/spectrum.csv".format(subdir=datum.name)
        for datum in metadata.itertuples()
        if datum.use_in_Ls_scan
    ]


rule collate_mres_scan_largeLs:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=mres_scan_largeLs_data,
        metadata=config["metadata"],
    output:
        csv="intermediary_data/mres_scan_largeLs.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


rule mres_scan_largeLs:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/plot_mres_scan_different_betas.py",
        data=rules.collate_mres_scan_largeLs.output.csv,
        plot_styles=config["plot_styles"],
    output:
        plot="assets/plots/mres_scan_largeLs.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --data {input.data} --plot_styles {input.plot_styles} --output_file {output.plot} --height 3"
