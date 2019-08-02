import configparser


def get_config_value(func, section, option, default_value, is_check_required=False):
    try:
        value = func(section, option)
    except:
        value = default_value

    if is_check_required and value is None:
        raise ValueError("PLEASE FILL THIS FIELD: section - {}, option - {}".format(section, option))
    return value


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    dict_config = {}

    common_section = "COMMON"
    dict_config["experiment_name"] = get_config_value(config.get, common_section, "experiment_name", None)
    dict_config["manual_seed"] = get_config_value(config.getint, common_section, "manual_seed", 9)
    dict_config["workers"] = get_config_value(config.getint, common_section, "workers", 4)
    dict_config["gpu"] = get_config_value(config.get, common_section, "gpu", None)
    dict_config["saved_models"] = get_config_value(config.get, common_section, "saved_models", None)

    train_section = "TRAIN"
    dict_config["batch_size"] = get_config_value(config.getint, train_section, "batch_size", 20)
    dict_config["epochs"] = get_config_value(config.getint, train_section, "epochs", 100)
    dict_config["num_iter"] = get_config_value(config.getint, train_section, "num_iter", 100)
    dict_config["print_interval"] = get_config_value(config.getint, train_section, "print_interval", 2)
    dict_config["continue_model"] = get_config_value(config.get, train_section, "continue_model", None)
    dict_config["optimizer"] = get_config_value(config.get, train_section, "optimizer", "adam")
    dict_config["beta1"] = get_config_value(config.getfloat, train_section, "beta1", 0.9)
    dict_config["eps"] = get_config_value(config.getfloat, train_section, "eps", 1e-8)
    dict_config["learning_rate"] = get_config_value(config.getfloat, train_section, "learning_rate", 0.1)
    dict_config["rho"] = get_config_value(config.getfloat, train_section, "rho", 0.95)
    dict_config["grad_clip"] = get_config_value(config.getint, train_section, "grad_clip", 5)
    dict_config["weight_decay"] = get_config_value(config.getfloat, train_section, "weight_decay", 1e-5)

    data_section = "DATA"
    dict_config["data_zip_file"] = get_config_value(config.get, data_section, "data_zip_file", None)
    dict_config["preprocess_data_name"] = get_config_value(config.get, data_section, "preprocess_data_name", None)
    dict_config["train_data_folder"] = get_config_value(config.get, data_section, "train_data_folder", None)
    dict_config["train_data_file"] = get_config_value(config.get, data_section, "train_data_file", None)
    dict_config["train_data_tgt"] = get_config_value(config.get, data_section, "train_data_tgt", None)
    dict_config["valid_data_file"] = get_config_value(config.get, data_section, "valid_data_file", None)
    dict_config["valid_data_tgt"] = get_config_value(config.get, data_section, "valid_data_tgt", None)
    dict_config["input_channel_size"] = get_config_value(config.get, data_section, "input_channel_size", 0)
    dict_config["sequence_length"] = get_config_value(config.getint, data_section, "sequence_length", 100)
    dict_config["sensitive"] = get_config_value(config.getboolean, data_section, "sensitive", True)
    dict_config["vocab_size"] = get_config_value(config.getboolean, data_section, "vocab_size", 1000)
    dict_config["sequence_length"] = get_config_value(config.getboolean, data_section, "sequence_length", 150)
    dict_config["tgt_words_min_frequency"] = get_config_value(config.get, data_section, "tgt_words_min_frequency", 1)
    dict_config["shard_size"] = get_config_value(config.get, data_section, "shard_size", 500)
    dict_config["world_size"] = get_config_value(config.get, data_section, "world_size", 2)
    dict_config["max_grad_norm"] = get_config_value(config.get, data_section, "max_grad_norm", 20)

    architecture_section = "ARCHITECTURE"
    dict_config["word_vec_size"] = get_config_value(config.getint, architecture_section, "word_vec_size", 80)
    dict_config["output_channel"] = get_config_value(config.getint, architecture_section, "output_channel", 512)
    dict_config["hidden_size"] = get_config_value(config.getint, architecture_section, "hidden_size", 512)
    dict_config["layers"] = get_config_value(config.getint, architecture_section, "layers", 1)
    dict_config["encoder_type"] = get_config_value(config.getint, architecture_section, "encoder_type", 'brnn')

    return dict_config
