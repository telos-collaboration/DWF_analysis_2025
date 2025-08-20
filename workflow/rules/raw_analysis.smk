def WF_logs(wildcards):
    return glob(f"{config['wf_dir']}/{wildcards.subdir}/wflow.*.out")


rule WF_and_Q:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/WF_and_Q.py",
        data=WF_logs,
    output:
        summary="intermediary_data/mdwf/{subdir}/WF_summary.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --tag {wildcards.subdir} --output_file {output.summary} --W0 {config[W0]}"


rule spectrum_and_decay:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
        L=lookup(query="name == '{wildcards.subdir}'", within=metadata, cols="Nx"),
    input:
        script="src/spectrum_and_decay.py",
        data=glob_wildcards(f"{config['correlator_dir']}/{{subdir}}/*.xml"),
        plot_styles=config["plot_styles"],
    output:
        csv="intermediary_data/mdwf/{subdir}/spectrum.csv",
        ZA_plot=f"intermediary_data/mdwf/{{subdir}}/ZA{config['plot_filetype']}",
        mres_plot=f"intermediary_data/mdwf/{{subdir}}/mres{config['plot_filetype']}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} --correlator_dir {config[correlator_dir]}/{wildcards.subdir} --csv_file {output.csv} --spatial_extent {params.L} --plot_styles {input.plot_styles} --output_file_mres {output.mres_plot} --output_file_ZA {output.ZA_plot}"


rule collate_eff_masses:
    input:
        plot="intermediary_data/mdwf/{ensemble}/{plot_name}.{filetype}",
    output:
        plot="assets/plots/effmass_{plot_name}_{ensemble}.{filetype}",
    shell:
        "cp {input.plot} {output.plot}"


def hmc_logs(wildcards):
    return glob(f"{config['hmc_dir']}/{wildcards.subdir}/hmc_*.out")


rule hmc_timing:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/hmc_timing.py",
        logs=hmc_logs,
    output:
        csv="intermediary_data/mdwf/{subdir}/timing.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.logs} --output_file {output.csv} --tag {wildcards.subdir}"
