rule assets_stamp:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/provenance.py",
        plots=plots,
    output:
        "assets/info.json",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.plots} --output_file {output}"


rule data_assets_stamp:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/provenance.py",
        CSVs=CSVs,
    output:
        "data_assets/info.json",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.plots} --output_file {output}"
