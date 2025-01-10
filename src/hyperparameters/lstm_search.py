import pytorch_lightning as pl
from ray import tune
from pytorch_lightning.callbacks import Callback
from train_ns_lstm import get_data, Model
from src.hyperparameters.search import main_tune
from src.args import init_lstmgnn_args, add_tune_params, add_configs


class ihm_TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics['val_loss'],
            acc=trainer.callback_metrics['acc'],
            auroc=trainer.callback_metrics['auroc'],
            auprc=trainer.callback_metrics['auprc'])

        
class los_TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics['val_loss'],
            mad=trainer.callback_metrics['mad'],
            mse=trainer.callback_metrics['mse'],
            mape=trainer.callback_metrics['mape'],
            msle=trainer.callback_metrics['msle'],
            r2=trainer.callback_metrics['r2'],
            kappa=trainer.callback_metrics['kappa'])


def main_train(config):
    dataset, train_loader, subgraph_loader = get_data(config)

    # define model
    model = Model(config, dataset, train_loader, subgraph_loader)

    trcb = [ihm_TuneReportCallback()] if config['task'] == 'ihm' else [los_TuneReportCallback()]

    trainer = pl.Trainer(
        gpus=config['gpus'],
        progress_bar_refresh_rate=0,
        weights_summary=None,
        max_epochs=config['epochs'],
        distributed_backend='dp',
        precision=16 if config['use_amp'] else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        callbacks=trcb
    )
    trainer.fit(model)



if __name__ == '__main__':
    parser = init_lstmgnn_args()
    parser = add_tune_params(parser)
    config = parser.parse_args()
    config.model = 'lstm'
    config = add_configs(config)
    main_tune(main_train, config)