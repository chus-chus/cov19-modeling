""" Params and paths for the preprocessing pipeline """

# paths
data_path = ''
data_path_incidence = ''

cleanWholeDataPath = ''
clean0To5DataPath = ''
cleanGT6DataPath = ''

# variables
incidence = False

# name of the columns to be used
selected_columns = ["date_of_birth", "simptomatology_date", "fever", "highest_fever", "total_days_fever", "date_fever",
                    "end_fever", "tos", "cough_first", "crup", "crup_first", "dysphonia", "disfonia_first",
                    "resp", "dyspnea_first", "tachypnea", "tachypnea_first", "ausc_resp", "auscult_first", "wheezing",
                    "crackles", "odynophagia",
                    "odynophagia_first", "nasal_congestion", "nasal_first", "fatiga", "fatigue_first", "headache",
                    "headache_first", "conjuntivitis",
                    "conj_first", "ocular_pain", "ocular_first", "gi_symptoms", "gi_first", "abdominal_pain",
                    "vomiting", "dyarrea", "dermatologic",
                    "skin_first", "rash", "inflam_periferic", "inflam_oral", "adenopathies", "lymph_first", "hepato",
                    "hepato_first", "splenomegaly",
                    "spleno_first", "hemorrhagies", "hemorr_first", "irritability", "irritability_first", "neuro",
                    "neuro_first", "confusion", "seizures",
                    "nuchal_stiffness", "hypotonia", "peripheral_paralysis", "shock", "shock_first", "taste_smell",
                    "taste_first", "smell", "smell_first",
                    "final_diagnosis_code"]

# name of attributes that indicate the presence of one symptom and are complementary to those on 'firsts'
normals = ["tos", "crup", "dysphonia",
           "resp", "tachypnea", "odynophagia",
           "nasal_congestion", "fatiga", "headache", "conjuntivitis",
           "ocular_pain", "adenopathies", "hepato", "splenomegaly",
           "hemorrhagies", "irritability", "shock", "taste_smell", "smell"]

# name of attributes that indicate if one symptom appeared during the first 48h
firsts = ["cough_first", "crup_first", "disfonia_first", "dyspnea_first", "tachypnea_first",
          "odynophagia_first", "nasal_first", "fatigue_first", "headache_first", "conj_first", "ocular_first",
          "lymph_first", "hepato_first", "spleno_first", "hemorr_first", "irritability_first", "shock_first",
          "taste_first", "smell_first"]

# variables used for the imputation of the target
targets = ['antigenic_result', 'pcr_result']

# columns remove for the different datasets
remove_all = ["crup", 'ocular_pain', 'hepato', 'splenomegaly']
remove_age0 = ['taste_smell', 'smell']
remove_age1 = ['irritability']
