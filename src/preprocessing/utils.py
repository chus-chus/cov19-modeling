import numpy as np
import pandas as pd


def read_data(data_path):
    """
  Compute age of the patient at symptomatology date.

  data_path: string
  """
    return pd.read_csv(data_path)


def compute_age(data):
    """
  Compute age of the patient at symptomatology date.

  data: pandas dataframe with column including "symptomatology_date" and "date_of_birth"
  """
    data["age"] = ((pd.to_datetime(data["simptomatology_date"]) - pd.to_datetime(
        data["date_of_birth"])).dt.days / 365).apply(np.floor)
    data.drop(["date_of_birth", "simptomatology_date"], axis=1, inplace=True)
    return data


def combine_normal_first(normal, first, x):
    """
  Combine "normal" column with its corresponding column of the same symptom
  indicating if it appear during the first 48h.

  normal: string name of the symptom column
  first: string name of the column first 48h
  x: row
  """
    if np.isnan(x[normal]) or x[normal] == 3:
        return -1  # unknown
    elif x[first] is not None and x[first] == 1:
        return 1  # first 48h
    elif x[normal] == 1 and x[first] == 0:
        return 2  # after 48h
    else:
        return 0  # no symptom


def group_days_fiver(days_fever, x):
    """
  Categorize fever variable in 3 different groups.

  days_fever: string name of the column
  x: row
  """
    if x[days_fever] == 1 or x[days_fever] == 2:
        return 1
    elif x[days_fever] >= 7:
        return 3
    elif 3 < x[days_fever] < 7:
        return 2
    else:
        return 0


def compute_fever(data):
    """
  Compute the total number of days with fever and categorize it.

  data: pandas dataframe
  """
    data["date_fever"] = pd.to_datetime(data["date_fever"], errors='coerce')
    data["end_fever"] = pd.to_datetime(data["end_fever"], errors='coerce')
    #  compute number of days
    data["total_days_fever"] = (data["end_fever"] - data["date_fever"]).dt.total_seconds() / (3600 * 24)
    data.drop(["date_fever", "end_fever"], axis=1, inplace=True)

    # infer category for special cases
    data.loc[data['total_days_fever'] < 0, 'total_days_fever'] = np.nan
    data.loc[data['fever'] == 0, 'total_days_fever'] = 0  # no fever
    data.loc[data['fever'] == -1, 'total_days_fever'] = -1  # unknown fever

    # infer the days with the median of the group
    filled = data.groupby('fever').transform(lambda x: x.fillna(x.median()))
    data['total_days_fever'] = filled['total_days_fever']

    # Categorize into groups
    data["total_days_fever"] = data.apply(lambda x: group_days_fiver("total_days_fever", x), axis=1)

    return data


def create_gi(normal, first, x):
    """
    Categorize gi attribute

    normal: string name of the gi column
    first: string name of the gi column indicating if it appear during the first 48h
    x: row
  """
    if np.isnan(x[normal]):
        return -1  # unknown
    elif x[first] == 1 and x[normal] == 1:
        return 1  # symptom during the first 48 h
    elif x[normal] == 1 and x[first] == 0:
        return 2  # symptom after the first 48 h
    elif x[normal] == 0:
        return 0  # no symptom


def compute_gi(data):
    """
  Reformat values of the categories for the gi attribute

  data: pandas dataframe
  """
    # Encode the appearance of the symptom during the first 48h as other category
    data["gi_symptoms"] = data.apply(lambda x: create_gi("gi_symptoms", "gi_first", x), axis=1)
    data.drop("gi_first", axis=1, inplace=True)
    # Recode values of categories
    cols = ['abdominal_pain', 'vomiting', 'dyarrea']
    data[cols] = data[cols].fillna(0)
    data['gi_symptoms'] = data['gi_symptoms'].fillna(2)
    data[cols] = data[cols].replace(2, 0)
    data[cols] = data[cols].replace(3, -1)
    return data


def formatting_fever(fever, highest_fever, x):
    """Reformat values of the categories for the fever attribute

  fever: string name of the fever column
  highest_fever: string name of the highest fever column
  x: row """
    if np.isnan(x[fever]):
        return -1  # unknown
    elif x[fever] == 2:
        return 0  # no symptom
    elif x[fever] == 1 and not np.isnan(x[highest_fever]):
        return x[highest_fever]  # >39
    else:
        return 2  # >38 a <=39


def create_neuro(normal, first, x):
    if np.isnan(x[normal]):
        return -1  # unknown
    elif x[normal] == 0:
        return 0  # no symptom
    elif x[normal] == 1 and x[first] == 1:
        return 1  # symptom during the first 48 h
    elif x[normal] == 1 and x[first] == 0:
        return 2  # symptom after the first 48 h


def compute_neuro(data):
    data["neuro"] = data.apply(lambda x: create_neuro("neuro", "neuro_first", x), axis=1)
    data.drop("neuro_first", axis=1, inplace=True)
    data.drop("confusion", axis=1, inplace=True)
    data.drop("seizures", axis=1, inplace=True)
    data.drop("nuchal_stiffness", axis=1, inplace=True)
    data.drop("hypotonia", axis=1, inplace=True)
    data.drop("peripheral_paralysis", axis=1, inplace=True)
    return data


def compute_derma(data):
    data["dermatologic"] = data.apply(lambda x: create_gi("dermatologic", "skin_first", x), axis=1)
    data.drop("skin_first", axis=1, inplace=True)
    data.drop("inflam_periferic", axis=1, inplace=True)
    data.drop("inflam_oral", axis=1, inplace=True)

    cols = ['rash']
    data[cols] = data[cols].fillna(0)
    data['dermatologic'] = data['dermatologic'].fillna(2)
    return data


def create_ausc(normal, first, x):
    if np.isnan(x[normal]):
        return -1
    elif x[first] == 1 and x[normal] == 2:
        return 1
    elif x[normal] == 2 and x[first] == 0:
        return 2
    elif x[normal] == 1:
        return 0


def compute_ausc(data):
    data["ausc_resp"] = data.apply(lambda x: create_ausc("ausc_resp", "auscult_first", x), axis=1)
    data.drop("auscult_first", axis=1, inplace=True)
    data['ausc_resp'] = data['ausc_resp'].fillna(2)
    return data


# 0 nada
# 1 wheezing
# 2 crackles
# 3 both
def create_ausc_type(ausc, wheez, crackl, x):
    if x[ausc] == 0 or x[ausc] == -1:
        return 0
    elif x[wheez] == 1 and x[crackl] == 1:
        return 3
    elif x[wheez] == 1 and x[crackl] == 2:
        return 1
    elif x[wheez] == 2 and x[crackl] == 1:
        return 2


def compute_ausc_type(data):
    data["wheezing"] = data.apply(lambda x: create_ausc_type("ausc_resp", "wheezing", "crackles", x), axis=1)
    data.drop("crackles", axis=1, inplace=True)
    data.rename(columns={'wheezing': 'ausc_type'}, inplace=True)
    data['ausc_resp'] = data['ausc_resp'].fillna(2)
    # crackles es NA y wheezing positivo por lo tanto ponemos que es 1 (wheezing)
    data['ausc_type'] = data['ausc_type'].fillna(1)
    return data


# We have inpute null code targets and supected (code=2) from pcr and atigenic results
def impute_results(code, pcr, antigenic, x):
    if np.isnan(x[code]):
        if not np.isnan(x[pcr]) or not np.isnan(x[antigenic]):
            if x[pcr] == 1 or x[antigenic] == 1:
                return 1
            elif x[pcr] == 2 or x[antigenic] == 2:
                return 3
    elif x[code] == 2:
        if not np.isnan(x[pcr]) or not np.isnan(x[antigenic]):
            if x[pcr] == 1 or x[antigenic] == 1:
                return 1
            elif x[pcr] == 2 or x[antigenic] == 2:
                return 3
    else:
        return x[code]


def impute_target(data):
    data["final_diagnosis_code"] = data.apply(
        lambda x: impute_results("final_diagnosis_code", "pcr_result", "antigenic_result", x), axis=1)
    # delete 10 rows with target value null
    data = data[data['final_diagnosis_code'].notna()]
    data['final_diagnosis_code'] = data['final_diagnosis_code'].replace(3, 0)
    # data[data.final_diagnosis_code==3]['final_diagnosis_code']=0
    data.drop('pcr_result', axis=1, inplace=True)
    data.drop('antigenic_result', axis=1, inplace=True)
    return data
