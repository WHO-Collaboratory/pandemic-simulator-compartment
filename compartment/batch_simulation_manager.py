from concurrent.futures import as_completed
from compartment.helpers import get_executor_class
from compartment.simulation_manager import SimulationManager

ExecutorClass = get_executor_class()
class BatchSimulationManager:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers

    def _chunked(self, iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def _single_run(self, model, params):
        import copy
        model_copy = copy.deepcopy(model)
        # override rates attributes
        for key, val in params.items():
            parts = key.split(".")
            if len(parts) == 1:
                setattr(model_copy, key, val)
            elif parts[0] == "intervention" and len(parts) == 3:
                _, intervention_id, field_name = parts
                model_copy.intervention_dict[intervention_id][field_name] = val
            else:
                raise ValueError(f"Invalid parameter key: {key}")
        return SimulationManager(model_copy).run_simulation()

    def run_batch(self, model, n_sims, param_list):
        results = [None] * n_sims
        batch_size = self.max_workers or 2
        for batch_indices in self._chunked(range(n_sims), batch_size):
            with ExecutorClass(max_workers=batch_size) as pool:
                futures = {
                    pool.submit(self._single_run, model, param_list[i]): i
                    for i in batch_indices
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
        return results