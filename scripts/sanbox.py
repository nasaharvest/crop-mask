from src.models import Model

if __name__ == "__main__":
    data_dir = "../data"
    model_name = "Togo"

    model = Model.load_from_checkpoint(f"{data_dir}/models/{model_name}.ckpt")
    model.save()
