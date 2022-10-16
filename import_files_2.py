


def readFiles_2(path, sample_size):
    import pandas as pd
    from pandas import json_normalize
    import glob
    import json
    files = glob.glob(path)
    dfs = []
    counter = 0
    #df = pd.DataFrame()
    for file in files[0:sample_size]:
        with open(file) as f:
            try:
                data = json.load(f)
                data = flatten_json(data)
                data = json_normalize(data)
                dfs.append(data)
                counter += 1
                print(counter)
            except ValueError:
                pass
    df = pd.concat(dfs, sort=False)
    return df

#Flatten .json files // forked from: "https://pypi.org/project/flatten-json/"
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            i = 0
            for a in x:
                #if i <= 2:
                flatten(a, name + 'l' + str(i+1) + '.')
                i += 1 #stops flattening of json after 1 layer
            
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def readFiles_threshold(path, threshold, key_list):
    import pandas as pd
    from pandas import json_normalize
    import glob
    import json
    files = glob.glob(path)
    key_list = key_list
    threshold = threshold
    dfs = []
    counter = 0
    ##create df of threshold values
    for file in files:
        with open(file) as f:
            try:
                data = json.load(f)
                data = flatten_json(data)
                if flatten_json(data)['result.receipts_outcome.l1.outcome.gas_burnt'] >= threshold:
                    data = json_normalize(data)
                    data = data[key_list]
                    dfs.append(data)
                    counter += 1
                    print(counter)
                    if counter == 50: #this is a testing variable that can be commented out. The algorithm will then read all threshold values in the db for testing.
                        break
            except ValueError:
                pass   
    df = pd.concat(dfs, sort=False)
    return df
