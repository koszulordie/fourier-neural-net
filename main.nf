
nextflow.enable.dsl=2

params.raw_input = "${projectDir}/data/INTERVAL_for_ferran.xlsx"


process PREPROCESS {
    container "docker.io/ferranmuinos/fourier-regression:latest"
    
    input:
    path input_file
    
    output:
    path "output.tsv"

    script:
    """
    python ${projectDir}/preprocess.py preprocess \
        --input_file "${input_file}" \
        --output_file "output.tsv"
    """
}


process GENERATE_RESPONSES {
    container "docker.io/ferranmuinos/fourier-regression:latest"

    input:
    path input
    
    output:
    path "responses.csv"

    script:
    """
    python ${projectDir}/preprocess.py responses \
        --input_file ${input} \
        --output_file responses.csv
    """
}


process FOURIER_REGRESSION {
    container "docker.io/ferranmuinos/fourier-regression:latest"
    publishDir "${projectDir}/output/${response}/", mode: 'copy'

    input:
    tuple val(response), path(preprocessed_data)

    output:
    path "out_model_${response}.keras"
    path "out_performance_${response}.png"
    path "out_shap_boxplots_${response}.png"
    path "out_partial_dependence_sex_${response}.png"
    path "out_partial_dependence_age_${response}.png"
    path "out_summary_${response}.pickle"

    script:
    """
    python ${projectDir}/fourier_regression.py \
        --response ${response} \
        --data ${preprocessed_data} \
        --out_model out_model_${response}.keras \
        --out_performance out_performance_${response}.png \
        --out_shap_boxplots out_shap_boxplots_${response}.png \
        --out_partial_dependence_sex out_partial_dependence_sex_${response}.png \
        --out_partial_dependence_age out_partial_dependence_age_${response}.png \
        --out_summary out_summary_${response}.pickle
    """

}



workflow {
    preprocessed_ch = PREPROCESS(params.raw_input)
    responses_ch = GENERATE_RESPONSES(preprocessed_ch)
    response_data_ch = responses_ch.splitCsv().map { item -> item[0] }.combine(preprocessed_ch)
    FOURIER_REGRESSION(response_data_ch)
}
