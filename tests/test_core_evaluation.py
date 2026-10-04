from dataclasses import replace
from bcs.calibration import calibration_panel,learn_core
from bcs.simulator import WorldSpec,Regime
from bcs.core_evaluation import compare_kernels,evaluate_core
from bcs.dataset import build_dataset,DatasetCounts


def test_all_joint_clamps_agree_with_learned_kernel_and_mutation_is_detected():
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    core=learn_core(calibration_panel(spec).probes)
    report=compare_kernels(core,spec)
    assert report['checked_transitions']==3888
    assert report['mismatches']==0
    assert compare_kernels(replace(core,y_table=(0,0,0,0)),spec)['mismatches']>0


def test_core_evaluation_preserves_category_counts_and_marks_small_run(tmp_path):
    build_dataset(tmp_path,DatasetCounts(8,6,6,1),seed=4)
    report=evaluate_core(tmp_path)
    assert report['histories']==6
    assert report['category_means']=={'predict_hold':0,'do_c_zero_hold':0,'do_c_one_hold':0,'replace_past_command':0}
    assert report['joint_mean_tv']==0
    assert report['oracle_validation_gate']=='not_run_registered_dataset'


def test_standalone_evaluator_rejects_fabricated_registered_dataset(tmp_path):
    import json
    from bcs.generator import Executed,event_wire
    from bcs.simulator import Action
    core=learn_core(calibration_panel(WorldSpec.for_regime(Regime.DETERMINISTIC)).probes)
    (tmp_path/'evaluator').mkdir()
    (tmp_path/'core.json').write_text(json.dumps(core.to_wire()))
    (tmp_path/'manifest.json').write_text('{"mode":"registered_size"}')
    row={'sides':[{'events':[event_wire(Executed(1,0,Action.START))]}]}
    (tmp_path/'evaluator/validation.jsonl').write_text((json.dumps(row)+'\n')*4000)
    report=evaluate_core(tmp_path)
    assert report['oracle_validation_gate']=='invalid_dataset'
