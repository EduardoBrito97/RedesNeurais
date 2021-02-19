import csv

results = []
with open('results_30hn.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    headerRow = next(csvReader)
    for row in csvReader:
        result = {}
        for value in headerRow:
            result[value] = row[headerRow.index(value)]
        results.append(result)

with open('results_60hn.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    headerRow = next(csvReader)
    for row in csvReader:
        result = {}
        for value in headerRow:
            result[value] = row[headerRow.index(value)]
        results.append(result)
