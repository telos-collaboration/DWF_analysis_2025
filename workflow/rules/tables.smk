rule collate_main_ensemble_results:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=lambda wildcards: log_collator(
            wildcards,
            [("use_in_main_analysis", True)],
            "intermediary_data/mdwf/{name}",
            ["spectrum.csv", "WF_summary.csv"],
        ),
        metadata=config["metadata"],
    output:
        csv="intermediary_data/main_results.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


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


rule tabulate_ensembles:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/tabulate_ensembles.py",
        spectrum_csv=rules.collate_main_ensemble_results.output.csv,
    output:
        tex="assets/tables/ensembles.tex",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.spectrum_csv} --output_file {output.tex}"


rule collate_finite_volume_ensemble_results:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/concatenate_csv.py",
        data=lambda wildcards: log_collator(
            wildcards,
            [("use_in_finite_volume_study", True), ("have_spectrum", True)],
            "intermediary_data/mdwf/{name}",
            ["spectrum.csv"],
        ),
        metadata=config["metadata"],
    output:
        csv="intermediary_data/finite_volume_results.csv",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.data} --metadata_file {input.metadata} --output_file {output.csv}"


rule tabulate_finite_volume_ensembles:
    params:
        module=lambda wildcards, input: input.script.replace("/", ".")[:-3],
    input:
        script="src/tabulate_finitevolume_ensembles.py",
        spectrum_csv=rules.collate_finite_volume_ensemble_results.output.csv,
    output:
        tex="assets/tables/finite_volume_ensembles.tex",
    conda:
        "../envs/basic-analysis.yml"
    shell:
        "python -m {params.module} {input.spectrum_csv} --output_file {output.tex} --sort_key beta --sort_key mF --sort_key Nt --sort_key Nx --sort_key Ls"
