import os
import sys

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver

from mnist_net import Net


if __name__ == '__main__':
    job_dir, bucket_name, mlflow_tracking_server = sys.argv[1:]

    n_clients = 3
    num_rounds = 10
    
    job = FedAvgJob(
        initial_model=Net(),
        n_clients=n_clients,
        num_rounds=num_rounds,
        name=f'fedavg-mnist',
        key_metric='accuracy',
    )

    cse_ctrl = CrossSiteModelEval(
        model_locator_id=job.comp_ids['locator_id'],
    )
    job.to_server(cse_ctrl)
    
    receiver = MLflowReceiver(
        tracking_uri=mlflow_tracking_server,
        kw_args={
            "experiment_name": "MNIST FLARE Experiment",
            "run_name": 'nvflare-mnist',
        },
    )
    job.to_server(receiver)

    train_script = "mnist_fl.py"

    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=f'--bucket-name {bucket_name}', framework=FrameworkType.PYTORCH, launch_external_process=True,
        )
        job.to(executor, f"site-{i + 1}")

    job.export_job(job_dir)
