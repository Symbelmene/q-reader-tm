import json
import pandas as pd


def read_and_parse_csv(csvPath):
    return pd.read_csv(csvPath)


def determine_view_point(key):
    if key['Scenario'] == 'T':
        return 'TrainingUO'
    if key['Within Participant'] == 'God\'s Eye':
        return 'GodsEye'
    if key['Within Participant'].lower() == 'Unit Only'.lower():
        return 'UnitOnly'
    if key['Within Participant'] == 'Shared Unit':
        return 'SharedUnit'
    print(key['Within Participant'])
    raise KeyError(f'Could not determine view point for key {key}')


def preprocess_and_relabel_keys(df_keys, df_scores):
    # Pivot entries to list
    resultList = []
    for idx, row in df_scores.iterrows():
        participant = row['Participant ID']
        condition = row['Condition']
        for col in df_scores.columns:
            if col != 'Participant ID' and col != 'Condition':
                question = col.split('_')[0]
                id = col.split('_')[1]
                resultList.append({
                    'Participant ID': participant,
                    'Participant': participant.split('_')[0],
                    'Condition': condition,
                    'Question': question,
                    'Legacy ID': id,
                    'Score': row[col]
                })
    legacy_key_mapping = {
        'T': 'Training',
        '1': '1300',
        '2': '1405',
        '3': '1515'
    }

    for result in resultList:
        # Find corresponding row in df_keys
        key = df_keys.loc[(df_keys['Participant'] == result['Participant']) &
                          (df_keys['Run time'] == legacy_key_mapping[result['Legacy ID']])]
        if len(key) != 1:
            raise KeyError(
                f'Could not find unique key for participant {result["Participant"]} and run time {result["Legacy ID"]}')

        key = key.iloc[0]
        result['Scenario'] = 'Scenario' + key['Scenario']
        result['Run time'] = key['Run time']
        result['View Point'] = determine_view_point(key)

        del result['Legacy ID']
    df = pd.DataFrame(resultList)
    return df


def concatenate_and_pivot_by_key(df, key):
    dfPivot = df.copy()

    # Concatenate question and scenario column
    dfPivot['Question'] = df['Question'].astype(str)
    dfPivot[key] = df[key].astype(str)
    dfPivot['Label'] = df['Question'] + '_' + df[key]

    # Pivot to give Label as column and Participant as row
    return dfPivot.pivot(index='Participant ID', columns='Label', values='Score')


def sort_rows_and_columns(df):
    # Sort by row index
    df['sorter'] = df.index
    df['key1'] = df['sorter'].apply(lambda s: s.split('_')[0][0])
    df['key2'] = df['sorter'].apply(lambda s: int(s.split('_')[0][1:]))
    df.sort_values(['key1', 'key2'], ascending=[True, True], inplace=True)
    del df['sorter']
    del df['key1']
    del df['key2']

    # Sort by column index
    df.reindex(sorted(df.columns, key=lambda c: (c.split('_')[0], c.split('_')[1])), axis=1)

    return df


def main():
    keysCSV = 'reformat_keys/participant_keys.csv'
    scoresCSV = 'reformat_keys/scores_reordered.csv'

    df_keys = read_and_parse_csv(keysCSV)
    df_scores = read_and_parse_csv(scoresCSV)

    df = preprocess_and_relabel_keys(df_keys, df_scores)

    # Pivot by scenario
    dfScenario = concatenate_and_pivot_by_key(df, 'Scenario')
    dfScenario = sort_rows_and_columns(dfScenario)
    dfScenario.to_csv('reformat_keys/scores_pivoted_by_scenario.csv')

    # Pivot by Run Time
    dfRunTime = concatenate_and_pivot_by_key(df, 'Run time')
    dfRunTime = sort_rows_and_columns(dfRunTime)
    dfRunTime.to_csv('reformat_keys/scores_pivoted_by_run_time.csv')

    # Pivot by View Point
    dfViewPoint = concatenate_and_pivot_by_key(df, 'View Point')
    dfViewPoint = sort_rows_and_columns(dfViewPoint)
    dfViewPoint.to_csv('reformat_keys/scores_pivoted_by_view_point.csv')


if __name__ == '__main__':
    main()