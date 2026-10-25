"""Unchanged final-paper fit with observational finite AdamW guards."""
from pathlib import Path
import argparse,json,sys,time,traceback
import torch
R=Path(__file__).resolve().parents[1]
import campaign_common as common
import campaign_train as trainer
from campaign_models import digest
ORIGINAL_ADAMW=torch.optim.AdamW
STOP_AT=None
OUT=None
def check_tensors(named, phase, step):
    values = [(k, v.detach()) for k, v in named if torch.is_tensor(v)]
    if not values:
        return
    # Separate scalar CPU optimizer steps from GPU tensors; one sync per device.
    devices = {str(v.device) for _, v in values}
    failed = []
    for device in devices:
        group = [(k, v) for k, v in values if str(v.device) == device]
        good = torch.stack([torch.isfinite(v).all() for _, v in group]).all()
        if not bool(good):
            failed.extend({'tensor': k, 'shape': list(v.shape),
                           'nonfinite': int((~torch.isfinite(v)).sum()),
                           'nan': int(torch.isnan(v).sum()), 'inf': int(torch.isinf(v).sum())}
                          for k, v in group if not bool(torch.isfinite(v).all()))
    if failed:
        common.write(OUT / 'NUMERICAL_FAILURE.json', {'UTC': common.utc(), 'attempted_step': step,
                     'phase': phase, 'tensors': failed, 'automatic_protocol_change': False})
        raise FloatingPointError(f'Nonfinite tensor at update {step}, {phase}; stop without protocol change')

class GuardedAdamW(ORIGINAL_ADAMW):
    """Same AdamW implementation, state and defaults; checks do not alter tensors."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.absolute_step = 0
        self.stage_first = 0
        self.stage_started = time.monotonic()
        self.records = 0

    def named_state(self):
        return [(f'{i}/{k}', v) for i, state in enumerate(self.state.values()) for k, v in state.items()]

    def named_parameters(self):
        return [(str(i), p) for i, p in enumerate(p for g in self.param_groups for p in g['params'])]

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        steps = {int(s['step']) for s in self.state.values()}
        assert len(steps) == 1
        self.absolute_step = self.stage_first = steps.pop()
        check_tensors(self.named_state(), 'loaded_optimizer', self.absolute_step)
        check_tensors(self.named_parameters(), 'loaded_weights', self.absolute_step)
        self.stage_started = time.monotonic()
        return result

    def step(self, closure=None):
        attempted = self.absolute_step + 1
        params = self.named_parameters()
        check_tensors([(k + '/gradient', p.grad) for k, p in params if p.grad is not None], 'pre_step_gradients', attempted)
        check_tensors(params + self.named_state(), 'pre_step_weights_optimizer', attempted)
        result = super().step(closure)
        check_tensors(params + self.named_state(), 'post_step_weights_optimizer', attempted)
        self.absolute_step = attempted
        if attempted % 64 == 0:
            elapsed = time.monotonic() - self.stage_started
            rate = elapsed / (attempted - self.stage_first)
            record = {'UTC': common.utc(), 'step': attempted, 'stage_first': self.stage_first,
                      'stage_wall_seconds': elapsed, 'wall_seconds_per_update': rate,
                      'ETA_to_gate_seconds': max(0, 128 - attempted) * rate,
                      'ETA_to_budget_seconds': max(0, 65536 - attempted) * rate,
                      'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                      'every_update_optimizer_gradient_weight_checks': 'all_finite',
                      'optimizer_parameter_states': len(self.state)}
            with (OUT / 'finite_guard.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            common.write(OUT / 'NUMERIC_PROGRESS.json', record)
            print('GUARDED_PROGRESS', json.dumps(record), flush=True)
        if STOP_AT is not None and attempted == STOP_AT:
            # Original trainer performs prescribed tuning and saves full RNG/Adam
            # checkpoint before responding to its existing clean-stop mechanism.
            common.STOP = True
        return result


class ActiveGuardedAdamW(GuardedAdamW):
    pass

def main(args):
    global STOP_AT, OUT
    STOP_AT=args.stop_at
    OUT=R/'results/restoration'/args.run
    for relative,expected in common.read(R/'audit/SCIENCE_FILE_SHA256.json')['files'].items():
        assert digest(R/relative)==expected,relative
    torch.optim.AdamW=ActiveGuardedAdamW
    common.STOP=False
    try:
        return trainer.run(argparse.Namespace(run=args.run, protocol=str(R/'config/CAMPAIGN_v1.json'),
            data_root=common.read(R/'audit/DATA_PREFLIGHT.json')['data_root'],results_root=str(R/'results')))
    except BaseException as exc:
        common.write(OUT/'RUN_EXCEPTION.json',{'UTC':common.utc(),'exception':repr(exc),'traceback':traceback.format_exc(),'automatic_retry':False,'protocol_changed':False})
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stop-at',type=int)
    raise SystemExit(main(p.parse_args()))
