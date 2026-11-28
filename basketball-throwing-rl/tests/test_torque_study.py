from scripts.torque_study import make_protocol, validate_initializations
import pytest


def test_protocol_is_predeclared_factorial_seeded_and_random_goals():
    p = make_protocol(1000000, [42, 43, 44])
    assert p == make_protocol(1000000, [42, 43, 44])
    assert p['conditions'] == {'none': [0., 0.], 'torque': [.5, 0.], 'work': [0., .02], 'both': [.5, .02]}
    assert len(p['targets_xz']) == 300
    assert len({tuple(t) for t in p['targets_xz']}) == 300
    assert all(1 <= x <= 1.5 and .75 <= z <= 1.1 for x,z in p['targets_xz'])
    assert p['checkpoint_steps'] == [100000, 250000, 500000, 750000]
    with pytest.raises(ValueError):
        make_protocol(1000000, [42, 42, 44])
    with pytest.raises(ValueError):
        make_protocol(1000000, [42, 43])


def test_initialization_audit_rejects_warm_start_and_unpaired_weights():
    rows=[{'seed':s,'initialization':{'mode':'from_scratch','num_timesteps':0,'observation_count':.0001,'policy_sha256':str(s)}} for s in [42,43,44] for _ in range(4)]
    validate_initializations(rows)
    rows[0]['initialization']['num_timesteps']=1000000
    with pytest.raises(ValueError):
        validate_initializations(rows)
    rows[0]['initialization']['num_timesteps']=0
    rows[0]['initialization']['policy_sha256']='different'
    with pytest.raises(ValueError):
        validate_initializations(rows)
