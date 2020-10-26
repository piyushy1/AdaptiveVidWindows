import re

# simplified VEQL query strucutre below accuracy 2 means top-2
query1 = 'CONJ(car,person) WITHINWINDOW(10,5) ACCURACY=2'

# a rigid and simplified query parser
def parse_query():
    query_predicates = {}
    predicates = re.split(' ', query1)
    for i in range(0, len(predicates)):
        if i ==0:
            split_pred = re.split( '[(,)]', predicates[i])
            del split_pred[-1] # remove the last element as it contains an extra quotes
            query_predicates['operator'] = split_pred[0]
            query_predicates['object'] = split_pred[1:]
            #query_predicates['object2'] = split_pred[2]
            #print(split_pred)
        if i ==1:
            split_pred = re.split('[(,)]', predicates[i])
            query_predicates['RANGE'] = int(split_pred[1])
            query_predicates['SLIDE'] = int(split_pred[2])
        if i ==2:
            split_pred = re.split('\=', predicates[i])
            query_predicates[split_pred[0]] = int(split_pred[1])
            #print(split_pred)

    return query_predicates

#parse_query()