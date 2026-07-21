import mlflow
import mlflow.pytorch


def setup_mlflow():

    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("two_tower_recsys")


def log_params(embedding_dim, batch_size, learning_rate):

    mlflow.log_param("embedding_dim", embedding_dim)

    mlflow.log_param("batch_size", batch_size)

    mlflow.log_param("learning_rate", learning_rate)


def log_metrics(
    epoch,
    train_loss,
    valid_loss,
    valid_auc,
    recall_at_k=0.0,
    ndcg_at_k=0.0,
):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("valid_loss", valid_loss, step=epoch)
    mlflow.log_metric("valid_auc", valid_auc, step=epoch)
    mlflow.log_metric("valid_recall_at_10", recall_at_k, step=epoch)
    mlflow.log_metric("valid_ndcg_at_10", ndcg_at_k, step=epoch)


def log_model(model):

    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")
