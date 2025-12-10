import huggingface_hub
import gin
import util
import dataset
import train
import evaluate

@gin.configurable
def main(hug_token, model_dir, model_name, supervised, finetuning):
    # log into huggingface with key for automatic download of models
    huggingface_hub.login(hug_token)

    output_dir = util.create_output_dir(dir=model_dir, model_name=model_name)

    if finetuning:
        # dataset thing
        train_dataset, eval_dataset = dataset.get_dataset(model_name=model_name, supervised=supervised)

        # training thing
        train.train(model_name=model_name, train_dataset=train_dataset, eval_dataset=eval_dataset,
                    supervised=supervised, output_dir=output_dir)

    # evaluation thing
    result = evaluate.evaluate(model_name=model_name, output_dir=output_dir, from_finetuned=finetuning)

    print(result)

if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['/home/florian/final/uvl-finetuning/config.gin'],[])
    main()
