import os

def create_output_dir(dir, model_name):
            output_dir = os.path.join(dir, model_name.split('/')[-1])
            return output_dir