"""Known-graph deterministic table learning from visible calibration histories."""
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
import json
from .simulator import Action, Variable as V, DeviceState, NoiseLaw, Step, Clamp, simulate, bit, object_id
from .inference import Evidence, History

ACTIONS = (Action.START,Action.STOP,Action.HOLD)

@dataclass(frozen=True)
class CalibrationProbe:
    target: V
    history: History
    output_time: int
    object_id: int = 0
    def __post_init__(self):
        object_id(self.object_id)
        if self.target not in (V.R,V.M,V.Y) or type(self.output_time) is not int or not 1 <= self.output_time <= len(self.history.steps):
            raise ValueError('invalid calibration probe')

@dataclass(frozen=True)
class CalibrationPanel:
    probes: tuple[CalibrationProbe,...]
    @property
    def action_count(self):
        return sum(len(p.history.steps) for p in self.probes)

@dataclass(frozen=True)
class Cell:
    target: V
    parent_index: int

@dataclass(frozen=True)
class IncompleteCalibration:
    missing_cells: tuple[Cell,...]

@dataclass(frozen=True)
class ConflictingCalibration:
    cell: Cell
    values: tuple[int,...]

@dataclass(frozen=True)
class BooleanCore:
    r_table: tuple[int,...]
    m_table: tuple[int,...]
    y_table: tuple[int,...]
    def __post_init__(self):
        for table,size in ((self.r_table,6),(self.m_table,2),(self.y_table,4)):
            if not isinstance(table,tuple) or len(table) != size:
                raise ValueError('complete immutable truth tables required')
            for value in table:
                bit(value)
    @property
    def noise(self):
        return NoiseLaw((F(1),F(0),F(0),F(0)))
    def reset(self,c,nm,ny):
        bit(c); bit(nm); bit(ny)
        m = self.m_table[0] ^ nm
        return DeviceState(0,m,c,self.y_table[2*m+c] ^ ny)
    def advance(self,previous,action,clamps,nm,ny):
        if not isinstance(action,Action):
            raise ValueError('typed action required')
        bit(nm); bit(ny)
        values = dict(clamps)
        if len(values) != len(clamps):
            raise ValueError('duplicate local clamp')
        for variable,value in clamps:
            Clamp(0,variable,value)
        c = values.get(V.C,previous.c)
        r = values.get(V.R,self.r_table[3*previous.r+ACTIONS.index(action)])
        m = values.get(V.M,self.m_table[r] ^ nm)
        y = values.get(V.Y,self.y_table[2*m+c] ^ ny)
        return DeviceState(r,m,c,y)
    def to_wire(self):
        return {'schema':'boolean-core-v1','r_table':list(self.r_table),'m_table':list(self.m_table),'y_table':list(self.y_table),
                'assumptions':['known_graph','deterministic','C_copy','R_reset_zero','independent_objects','stationary_mechanisms']}
    @classmethod
    def from_wire(cls,value):
        if not isinstance(value,dict) or set(value) != {'schema','r_table','m_table','y_table','assumptions'} or value['schema'] != 'boolean-core-v1':
            raise ValueError('invalid core artifact')
        core = cls(tuple(value['r_table']),tuple(value['m_table']),tuple(value['y_table']))
        if value['assumptions'] != core.to_wire()['assumptions']:
            raise ValueError('unsupported core assumptions')
        return core
    @property
    def artifact_id(self):
        return 'sha256:'+sha256(json.dumps(self.to_wire(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def calibration_panel(spec,seed=0):
    """Evaluator collects a fixed panel. Only returned visible probes train core."""
    if spec.noise.weights != (F(1),F(0),F(0),F(0)):
        raise ValueError('deterministic calibration only')
    probes = []
    def collect(target,steps,measurements):
        trace = simulate(spec,f'calibration-{len(probes)}',seed,steps)
        evidence = tuple(Evidence(time,0,var,trace.states[time][0].get(var)) for time,var in measurements)
        probes.append(CalibrationProbe(target,History(steps,evidence),len(steps)))
    for r in (0,1):
        for action in ACTIONS:
            collect(V.R,(Step((Action.HOLD,Action.HOLD),(Clamp(0,V.R,r),)),Step.command(0,action)),((1,V.R),(2,V.R)))
    for r in (0,1):
        collect(V.M,(Step((Action.HOLD,Action.HOLD),(Clamp(0,V.R,r),)),),((1,V.R),(1,V.M)))
    for m in (0,1):
        for c in (0,1):
            collect(V.Y,(Step((Action.HOLD,Action.HOLD),(Clamp(0,V.C,c),)),Step((Action.HOLD,Action.HOLD),(Clamp(0,V.M,m),))),((2,V.M),(2,V.C),(2,V.Y)))
    return CalibrationPanel(tuple(probes))


def learn_core(probes):
    """No source model, hidden trace, simulator U, or test data enters this API."""
    cells = {}
    for probe in probes:
        t,obj = probe.output_time,probe.object_id
        step = probe.history.steps[t-1]
        if any(c.object_id == obj and c.variable == probe.target for c in step.clamps):
            raise ValueError('clamped child cannot teach its natural mechanism')
        def observed(time,var):
            values = {e.value for e in probe.history.evidence if e.time == time and e.object_id == obj and e.variable == var}
            if len(values) != 1:
                raise ValueError('each parent/child needs one unambiguous visible observation')
            return values.pop()
        if probe.target == V.R:
            index = observed(t-1,V.R)*3 + ACTIONS.index(step.actions[obj])
        elif probe.target == V.M:
            index = observed(t,V.R)
        else:
            index = 2*observed(t,V.M)+observed(t,V.C)
        cell = Cell(probe.target,index)
        cells.setdefault(cell,set()).add(observed(t,probe.target))
        if len(cells[cell]) > 1:
            return ConflictingCalibration(cell,tuple(sorted(cells[cell])))
    required = tuple(Cell(var,i) for var,size in ((V.R,6),(V.M,2),(V.Y,4)) for i in range(size))
    missing = tuple(cell for cell in required if cell not in cells)
    if missing:
        return IncompleteCalibration(missing)
    values = [next(iter(cells[cell])) for cell in required]
    return BooleanCore(tuple(values[:6]),tuple(values[6:8]),tuple(values[8:]))
