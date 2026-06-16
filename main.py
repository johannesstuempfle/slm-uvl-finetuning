import huggingface_hub
import gin
import os
import src.util as util
import src.dataset as dataset
import src.train as train
import src.evaluate as evaluate
from pathlib import Path

@gin.configurable
def main(hug_token, model_dir, model_name, supervised, finetuning, adapter_dir=None, cuda_visible_devices=None):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    # log into huggingface with key for automatic download of models
    huggingface_hub.login(hug_token)

    if adapter_dir:
        output_dir = adapter_dir
    else:
        output_dir = util.create_output_dir(dir=model_dir, model_name=model_name)

        if finetuning:
            # dataset thing
            train_dataset, eval_dataset = dataset.get_dataset(model_name=model_name, supervised=supervised)

            # training thing
            train.train(model_name=model_name, train_dataset=train_dataset, eval_dataset=eval_dataset,
                        supervised=supervised, output_dir=output_dir)

    # evaluation thing
    result = evaluate.evaluate(model_name=model_name, output_dir=output_dir,
                               from_finetuned=finetuning or bool(adapter_dir))

    print(result)

if __name__ == "__main__":
    config_path = Path(__file__).parent / 'config.gin'
    gin.parse_config_files_and_bindings([str(config_path)], [])
    main()
