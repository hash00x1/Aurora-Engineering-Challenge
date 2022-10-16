def readFiles_2(path):
    import pandas as pd
    from pandas import json_normalize
    import glob
    import json
    files = glob.glob(path)
    dfs = []
    counter = 0
    #df = pd.DataFrame()
    for file in files[0:10000]:
        with open(file) as f:
            try:
                data = json.load(f)
                data = flatten_json(data)
                data = json_normalize(data)
                #for i in data:
                    #if data[i].dtype == object:
                        #print(i)
                        #data[i] = data[i].str.replace(r'\D', '', regex=True)
                        #data[i] = data[i].str.lstrip('0')
                        #data[i] = data[i].str.replace(r'^\s*$', '0', regex=True)
                        #data[i] = data[i].fillna(0)
                        #data[i] = data[i].astype('category')
                dfs.append(data)
                counter += 1
                print(counter)
            except ValueError:
                pass
    df = pd.concat(dfs, sort=False)
    return df

#Flatten .json files // forked from https://pypi.org/project/flatten-json/

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
                    if counter == 50:
                        break
            except ValueError:
                pass   
    df = pd.concat(dfs, sort=False)
    return df