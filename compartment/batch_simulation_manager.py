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
        # Rebuild the model from an overridden config rather than mutating a
        # constructed instance. Re-running __init__ re-derives every value
        # declared via add_transmission_edge / add_disease_parameter /
        # add_intervention, so uncertainty sampling covers params that the
        # model bakes into derived constants (e.g. 1/latent_period) or into
        # frozen Intervention objects.
        import copy

        overridden = model.build_overridden_config(params)
        if overridden is None:
            # Non-reconstructable model: fall back to copying the instance.
            model_copy = copy.deepcopy(model)
        else:
            model_copy = type(model)(overridden)
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