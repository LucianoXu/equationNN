# the proof traces
from __future__ import annotations
from .env import env, Scenario

class ProofTrace:
    def __init__(self, scenario: Scenario, trace: list[env.proof_step], final_stt: env.proof_state):
        self.scenario = scenario
        self.trace = trace
        self.final_stt = final_stt

    def __str__(self):
        res = "\n".join([str(step) for step in self.trace]) + "\n"
        res += "Final equation: " + str(self.final_stt) + "\n"
        return res
    
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
        