def retr_relations(facts: list, relation_dict):
    '''
    retriveve all the relations within a fact list like ['Q596874\P155\Q3205815', 'Q596874\P123\Q921536', 'Q596874\P136\Q1543778']
    return relation_id and relation_names in list format
    '''

    relation_ids = set()
    relation_names = set()
    for fact in facts:
        id = int(fact.split('|')[1])
        relation_ids.add(id)
        try:
            name = relation_dict[id]
            relation_names.add(name)
        except:
            pass

    return list(relation_names), list(relation_ids)

# def fact_triplet_to_sentence(triplet, relation_dict):
#     subject_id, predicate_id, object_id = triplet.split('\t')
#
#     # subject = id2entity(subject_id)
#     predicate = relation_dict[predicate_id]
#     # object_ = id2entity(object_id)
#
#     sentence = f"{subject} {predicate} {object_}"
#
#     return sentence