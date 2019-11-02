import os

def  make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def save_model_with_norm(model, env, path):
    model.save(path)
    path += "_running_avg"
    make_dir(path)
    env.save_running_average(path)

def load_model_with_norm(model_cls, env, path):
    env.load_running_average(path + "_running_avg")
    model = model_cls.load(path, env=env)
    return model