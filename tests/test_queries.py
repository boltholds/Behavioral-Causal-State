from bcs.queries import decode_query_ast
from bcs.inference import Interventional, Counterfactual
from bcs.simulator import Action, Variable as V, Clamp


def test_hard_do_wire_preserves_target_and_replacement_semantics():
    query = {'schema_version':'query-v1','kind':'Interventional','object_id':'device-0','outcome':['Y'],'horizon':1,'evidence_ref':'h','interventions':[{'object_id':'device-1','variable':'M','value':0}]}
    assert decode_query_ast(query) == Interventional(0,(Clamp(1,V.M,0),),(V.Y,))


def test_cf_wire_requires_explicit_factual_continuation_policy():
    query = {'schema_version':'query-v1','kind':'Counterfactual','object_id':'device-0','outcome':['Y'],'horizon':0,'evidence_ref':'h','replaced_time':1,'replacement':'stop','continuation_policy':'preserve_factual_actions'}
    assert decode_query_ast(query) == Counterfactual(0,1,Action.STOP,(V.Y,))
