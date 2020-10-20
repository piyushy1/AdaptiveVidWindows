import re


# a rigid and simplified query parser
def parse_query(query):
    query_predicates = {}
    predicates = re.split(' ', query)
    for i in range(0, len(predicates)):
        if i ==0:
            split_pred = re.split( '[(,)]', predicates[i])
            query_predicates['operator'] = split_pred[0]
            query_predicates['object1'] = split_pred[1]
            query_predicates['object2'] = split_pred[2]
            #print(split_pred)
        if i ==1:
            split_pred = re.split('[(,)]', predicates[i])
            query_predicates['RANGE'] = split_pred[1]
            query_predicates['SLIDE'] = split_pred[2]
        if i ==2:
            split_pred = re.split('\=', predicates[i])
            query_predicates[split_pred[0]] = split_pred[1]
            #print(split_pred)

    return query_predicates

#parse_query(query1)