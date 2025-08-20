def fig_label(wildcards):
    metadatum = metadata.query(f"name == {wildcards.subdir}")
    assert len(metadatum) == 1
    return f"$\\beta={metadatum['beta'][0]}, am={metadatum['mF'][0]}$"


rule plot_wflow_E:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
        label=fig_label,
    input:
        script="src/plot_WF_E.py",
        data=WF_logs,
        plot_styles=config["plot_styles"],
    output:
        plot="intermediary_data/mdwf/{subdir}/E_flow.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --plot_styles {input.plot_styles} --output_file {output.plot} --ensemble_label '{params.label}'"


rule plot_wflow_W:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
        label=fig_label,
    input:
        script="src/plot_WF_W.py",
        data=WF_logs,
        plot_styles=config["plot_styles"],
    output:
        plot="intermediary_data/mdwf/{subdir}/W_flow.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --plot_styles {input.plot_styles} --output_file {output.plot} --ensemble_label '{params.label}'"


rule plot_topological_charge:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
        label=fig_label,
    input:
        script="src/plot_topcharge.py",
        data=WF_logs,
        plot_styles=config["plot_styles"],
    output:
        plot="intermediary_data/mdwf/{subdir}/Q_history.{filetype}",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --plot_styles {input.plot_styles} --output_file {output.plot} --W0 {config[W0]} --ensemble_label '{params.label}'"
