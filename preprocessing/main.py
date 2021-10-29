""" Main preprocessing pipeline """

import pandas as pd

from preprocessing.configs import remove_all, remove_age0, remove_age1, targets, normals, firsts, data_path, \
    selected_columns, incidence, data_path_incidence, cleanWholeDataPath, clean0To5DataPath, cleanGT6DataPath
from preprocessing.utils import combine_normal_first, formatting_fever, compute_fever, compute_gi, compute_neuro, \
    compute_derma, compute_ausc, compute_ausc_type, impute_target, read_data, compute_age

if __name__ == '__main__':

    data = read_data(data_path)
    # select relevant columns
    symptoms_data = data[selected_columns].copy()
    # create age column
    symptoms_data = compute_age(symptoms_data)

    if incidence:
        data_inc = read_data(data_path_incidence)
        symptoms_data["incidence"] = data_inc["incidencia"]
        symptoms_data["incidence"] = symptoms_data["incidence"].fillna(-1)

    # combine normals with firsts
    for i in range(len(normals)):
        symptoms_data[normals[i]] = symptoms_data.apply(lambda x: combine_normal_first(normals[i], firsts[i], x),
                                                        axis=1)
        symptoms_data.drop(firsts[i], axis=1, inplace=True)
        symptoms_data[normals[i]] = symptoms_data[normals[i]].fillna(2)

    # Reformat fever column
    symptoms_data['fever'] = symptoms_data.apply(lambda x: formatting_fever('fever', 'highest_fever', x), axis=1)
    symptoms_data.drop('highest_fever', axis=1, inplace=True)

    # Create new columns for variables
    symptoms_data = compute_fever(symptoms_data)
    symptoms_data = compute_gi(symptoms_data)
    symptoms_data = compute_neuro(symptoms_data)
    symptoms_data = compute_derma(symptoms_data)
    symptoms_data = compute_ausc(symptoms_data)
    symptoms_data = compute_ausc_type(symptoms_data)
    # impute target: this imputation is dependant only on the observation itself
    df_target = data[targets]
    data_total = pd.concat([symptoms_data, df_target], axis=1)
    data_total = impute_target(data_total)
    data_total.drop(remove_all, axis=1, inplace=True)

    df_age0 = data_total[data_total['age'] <= 5]
    df_age0.drop(remove_age0, axis=1, inplace=True)

    df_age1 = data_total[data_total['age'] >= 6]
    df_age1.drop(remove_age1, axis=1, inplace=True)

    data_total.drop(["age"], axis=1, inplace=True)
    df_age0.drop(["age"], axis=1, inplace=True)
    df_age1.drop(["age"], axis=1, inplace=True)

    data_total.to_csv(cleanWholeDataPath)
    df_age0.to_csv(clean0To5DataPath)
    df_age1.to_csv(cleanGT6DataPath)
