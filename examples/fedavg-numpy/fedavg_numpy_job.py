import sys

from nvflare import FedJob
from fedavg import FedAvg
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector



if __name__ == '__main__':
    job_dir = sys.argv[1]

    job = FedJob(name="fedavg-numpy")

    n_clients = 3
    num_rounds = 3

    persistor_id = job.to_server(NPModelPersistor(), "persistor")

    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
        persistor_id=persistor_id,
    )
    job.to_server(controller)
    job.to(IntimeModelSelector(key_metric="accuracy"), "server")

    train_script = "numpy_fl.py"

    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args="", framework=FrameworkType.NUMPY, launch_external_process=True,
        )
        job.to(executor, f"site-{i + 1}")

    job.export_job(job_dir)
