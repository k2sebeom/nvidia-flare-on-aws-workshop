import os
import torch

from nvflare.app_common.app_event_type import AppEventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_common.app_constant import DefaultCheckpointFileName

import mlflow
from mlflow.models.signature import infer_signature


class PTMlflowModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        exclude_vars=None,
        model=None,
        sample_input_size=None,
        global_model_file_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_global_model_file_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_file_full_name=None,
        filter_id: str = None,
    ):
        super().__init__(
            exclude_vars=exclude_vars,
            model=model,
            global_model_file_name=global_model_file_name,
            best_global_model_file_name=best_global_model_file_name,
            source_ckpt_file_full_name=source_ckpt_file_full_name,
            filter_id=filter_id,
        )
        self.sample_input_size = sample_input_size

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        super().save_model(ml, fl_ctx)
        # Log model to global run
        self.save_to_mlflow(self._ckpt_save_path, fl_ctx)
    
    def handle_event(self, event: str, fl_ctx: FLContext):
        super().handle_event(event, fl_ctx)
        if event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current model as the best model, or the global best model if available
            self.save_to_mlflow(self._best_ckpt_save_path, fl_ctx)

    def save_to_mlflow(self, save_path: str, fl_ctx: FLContext):
        save_dict = self.persistence_manager.to_persistence_dict()

        model_weights = save_dict['model']
        model = self.model
        model.load_state_dict(model_weights)
        model.eval()

        run_id = fl_ctx.get_prop('SERVER_MLFLOW_RUN_ID')
        artifact_path = os.path.join('server', os.path.basename(save_path))

        signature = None
        if self.sample_input_size is not None:
            X = torch.rand(self.sample_input_size)
            signature = infer_signature(X.numpy(), model(X).detach().numpy())

        with mlflow.start_run(run_id=run_id):
            mlflow.pytorch.log_model(
                torch.jit.script(model),
                artifact_path=artifact_path,
                signature=signature,
                pip_requirements=['torch', 'mlflow', 'cloudpickle', 'pandas'],
            )
