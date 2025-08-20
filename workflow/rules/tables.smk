rule collate_main_spectrum_results:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=lambda wildcards: log_collator(
            wildcards,
            [("use_in_main_analysis", True)],
            "intermediary_data/mdwf/{name}",
            ["spectrum.csv"],
        ),
        metadata=config["metadata"],
    output:
        csv="intermediary_data/main_spectrum.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


rule tabulate_spectrum:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/tabulate_spectrum.py",
        spectrum_csv=rules.collate_main_spectrum_results.output.csv,
    output:
        tex="assets/tables/spectrum.tex",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.spectrum_csv} --output_file {output.tex}"
