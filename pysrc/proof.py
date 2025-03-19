# the proof traces
from __future__ import annotations
from .env import env, Scenario

class ProofTrace:
    def __init__(self, scenario: Scenario, steps: list[env.proof_step], final_stt: env.proof_state):
        self.scenario = scenario
        self.steps = steps
        self.final_stt = final_stt

    def __len__(self):
        return len(self.steps)

    def __str__(self):
        res = "\n".join([str(step) for step in self.steps]) + "\n"
        res += "Final State: " + str(self.final_stt) + "\n"
        return res
    
    @property
    def init_stt(self) -> env.proof_state:
        if len(self.steps) == 0:
            return self.final_stt
        else:
            return self.steps[0].stt
    
    @staticmethod
    def from_acts(scenario: Scenario, init_stt: env.proof_state, acts: list[str]) -> ProofTrace:
        '''
        Create a proof trace from a list of actions.
        '''
        trace = []

        for act in acts:
            stt = env.proof_state(init_stt)
            if scenario.kernel.action_by_code(stt, act) != env.ACT_RESULT.SUCCESS:
                raise ValueError(f"Cannot apply the action {act} to the equation {stt}.")
            proof_act = env.parse_proof_action(act)
            if proof_act is None:
                raise ValueError(f"Cannot parse the proof action: {act}.")
            trace.append(env.proof_step(init_stt, proof_act))
            init_stt = stt

        return ProofTrace(scenario, trace, stt)
        