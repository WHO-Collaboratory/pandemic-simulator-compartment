from concurrent.futures import ProcessPoolExecutor, as_completed
from compartment.simulation_manager import SimulationManager

class BatchSimulationManager:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers

    def _single_run(self, model, params):
        # override rates attributes
        for key, val in params.items():
            parts = key.split(".")
            if len(parts) == 1:
                setattr(model, key, val)
            elif parts[0] == "intervention" and len(parts) == 3:
                _, intervention_id, field_name = parts
                model.intervention_dict[intervention_id][field_name] = val
            else:
                raise ValueError(f"Invalid parameter key: {key}")
        return SimulationManager(model).run_simulation()

    def run_batch(self, model, n_sims, param_list):
        results = [None] * n_sims
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._single_run, model, param_list[i]): i
                for i in range(n_sims)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results