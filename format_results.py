import pandas as pd
from collections import defaultdict


def reformatResults(resultsCSVPath):
    df = pd.read_csv(resultsCSVPath, dtype=str)
    participantList = []
    for name, group in df.groupby('Participant ID'):
        outDict = {}
        for idx, row in group.iterrows():
            questionMarkKey = f'Q{row["Question No."]}_{row["Mark"].upper()}'
            if questionMarkKey in outDict:
                print(f'WARNING: Duplicate question mark key {questionMarkKey} found for participant {name}. '
                      f'Only the first value will be used.')
                continue
            outDict[questionMarkKey] = row["Score(%)"]
        participantList.append(outDict)

    outDf = pd.DataFrame(participantList)
    outDf['Participant ID'] = df['Participant ID'].unique()
    outDf = outDf[['Participant ID'] + sorted(outDf.columns[:-1], key=lambda x: (float(x.split('_')[0][1:]), x.split('_')[1]))]
    outDf = outDf[['Participant ID'] + [c for c in outDf.columns[1:] if '.' not in c] + [c for c in outDf.columns[1:] if '.' in c]]
    outDf.to_csv('output/bar_results_reformatted.csv', index=False)


def main():
    resultsCSVPath = 'output/bar_results.csv'
    reformatResults(resultsCSVPath)


if __name__ == '__main__':
    main()