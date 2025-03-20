# the proof traces
from __future__ import annotations
from typing import Optional
from .env import env, Scenario
from abc import ABC, abstractmethod
import multiprocessing as mp
from tqdm import tqdm

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

class ProblemSet(ABC):
    '''
    The problem set.
    '''
    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def problems(self) -> list[env.Equation]:
        '''
        Get the list of problems.
        '''
        pass

    def __str__(self):
        res = "% [PROBLEM SET] \n"
        res += str(self.scenario.alg).replace("\n", "\n% ") + "\n"

        for eq in self.problems:
            res += str(eq) + "\n"

        return res
    
    def save(self, path: str):
        '''
        Save the problem set to a file.
        '''
        with open(path, 'w') as f:
            f.write(str(self))
        

class GenProblemSet(ProblemSet):
    '''
    The generated problem set, consists of a list of proof traces.
    Each proof trace is a sequence of proof steps, with the obvious initial state, and the final state is the problem to solve.
    '''
    def __init__(self, scenario: Scenario, traces: list[ProofTrace]):
        super().__init__(scenario)
        self.traces = traces

    def __len__(self):
        return len(self.traces)
    
    @property
    def problems(self):
        return [trace.final_stt.eq for trace in self.traces]
    
    @property
    def avg_sol_len(self) -> float:
        '''
        Calculate the average solution length.
        '''
        return sum([len(trace) for trace in self.traces]) / len(self.traces)
    
    def __getstate__(self):
        # Only pickle the traces. Remove scenario from state if it isn’t pickleable.
        state = {
            'traces': self.traces,
            'scenario': self.scenario
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    

class GenProblemFactory(ABC):
    '''
    The factory to generate problem sets. The factory can be modified by operators to generate different problem sets.
    '''
    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    @abstractmethod
    def spawn(self, count: int) -> GenProblemSet:
        '''
        Generate a problem set.
        '''
        pass

    @abstractmethod
    def __str__(self):
        pass

class UniqueExamplePipe(GenProblemFactory):
    '''
    A pipe that return the factory to generate unique examples.
    '''
    def __init__(self, factory: GenProblemFactory):
        super().__init__(factory.scenario)
        self.factory = factory

    def spawn(self, count: int) -> GenProblemSet:

        unique_problems : list[env.Equation] = []
        traces : list[ProofTrace] = []

        with tqdm(range(count), desc="UniqueExamplePipe", leave=False) as progress:
            while len(unique_problems) < count:
                problem_set = self.factory.spawn(count // 2 + 1)
                for trace in problem_set.traces:
                    if trace.final_stt.eq not in unique_problems:
                        unique_problems.append(trace.final_stt.eq)
                        traces.append(trace)
                        progress.update(1)
                    if len(unique_problems) >= count:
                        break

        return GenProblemSet(self.scenario, traces)
    
    def __str__(self):
        res = str(self.factory)
        res += "\n=[Unique Example Pipe()]=>"
        return res

class LengthRequestPipe(GenProblemFactory):
    '''
    A pipe that request the minimal length of the problem set.
    '''
    def __init__(self, factory: GenProblemFactory, min_len: int):
        super().__init__(factory.scenario)
        self.factory = factory
        self.min_len = min_len

    def spawn(self, count: int) -> GenProblemSet:

        traces : list[ProofTrace] = []

        with tqdm(range(count), desc="LengthRequestPipe", leave=False) as progress:
            while len(traces) < count:
                problem_set = self.factory.spawn(count)
                for trace in problem_set.traces:
                    if len(trace) >= self.min_len:
                        traces.append(trace)
                        progress.update(1)
                    if len(traces) >= count:
                        break

        return GenProblemSet(self.scenario, traces)
    
    def __str__(self):
        res = str(self.factory)
        res += f"\n=[Length Request Pipe(Min Len={self.min_len})]=>"
        return res


class MultiProcessPipe(GenProblemFactory):
    '''
    A pipe that generate problem sets using multiple processes.
    '''
    def __init__(self, factory: GenProblemFactory, nproc: Optional[int] = None):
        super().__init__(factory.scenario)
        self.factory = factory
        self.nproc : int = nproc if nproc is not None else mp.cpu_count()

    def spawn(self, count: int) -> GenProblemSet:

        with tqdm(desc=f"MultiProcessPipe({self.nproc})", leave=False) as pbar:
            
            # every worker generates (count // nproc + 1) problems
            chunk_size = count // self.nproc + 1

            with mp.Pool(self.nproc) as pool:
                results = [pool.apply_async(self.factory.spawn, (chunk_size,))
                            for _ in range(self.nproc)]
                # wait for all processes to finish
                sets = [r.get() for r in results]

        # merge the problem sets
        traces : list[ProofTrace] = [trace for s in sets for trace in s.traces]
        traces = traces[:count]
            
        return GenProblemSet(self.scenario, traces)        

    def __str__(self):
        res = str(self.factory)
        res += f"\n=[Multi Process Pipe(NProc={self.nproc})]=>"
        return res


    