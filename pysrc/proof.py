# the proof traces
from __future__ import annotations
from .env import env, Scenario

class ProofTrace:
    def __init__(self, scenario: Scenario, trace: list[env.proof_step], final_eq: env.Equation):
        self.scenario = scenario
        self.trace = trace
        self.final_eq = final_eq

    def __str__(self):
        res = "\n".join([str(step) for step in self.trace]) + "\n"
        res += "Final equation: " + str(self.final_eq) + "\n"
        return res
    
    @staticmethod
    def from_acts(scenario: Scenario, init_eq: env.Equation, acts: list[str]) -> ProofTrace:
        '''
        Create a proof trace from a list of actions.
        '''
        trace = []

        for act in acts:
            eq = env.Equation(init_eq)
            if scenario.kernel.action_by_code(eq, act) != env.ACT_RESULT.SUCCESS:
                raise ValueError(f"Cannot apply the action {act} to the equation {eq}.")
            proof_act = env.parse_proof_action(act)
            if proof_act is None:
                raise ValueError(f"Cannot parse the proof action: {act}.")
            trace.append(env.proof_step(init_eq, proof_act))
            init_eq = eq

        return ProofTrace(scenario, trace, eq)
        