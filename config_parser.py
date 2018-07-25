import os
import json
import argparse


def get_value(dictionary, key_list):
    """
    dictionary = {"hoge": {"fuga": "boo"}}
    key_list["hoge", "fuga"]
    >> get_value(dictionary, key_list)
    "boo"
    """
    if len(key_list) == 1:
        return dictionary[key_list[0]]
    key = key_list.pop(0)
    return get_value(dictionary[key], key_list)


def __parse(org_dictionary, dictionary=None, variables=dict()):
    if dictionary is None:
        dictionary = org_dictionary

    new_dict = dict()
    for key, val in dictionary.items():
        if key == "__overwrite":
            key_list = val.strip().split(" ")
            value = get_value(org_dictionary, key_list)
            if isinstance(value, str):
                for src, tar in variables.items():
                    if src in value:
                        value = value.replace(src, tar)

            return value

        if isinstance(val, str):
            # replacement
            for src, tar in variables.items():
                if src in val:
                    val = val.replace(src, tar)

            new_dict[key] = val
        elif isinstance(val, dict):
            new_dict[key] = __parse(org_dictionary, dictionary=val, variables=variables)
        else:
            new_dict[key] = val
    return new_dict


def parse(config):
    # set variables if key "__variables" exits
    if "__variables" in config:
        model_name = config["__variables"]["___model_name___"]
        for value in config["__variables"].values():
            if value == "___model_name___": continue
            value.replace("___model_name___", model_name)
        variables = config["__variables"]
        del config["__variables"]
    else:
        variables = dict()
    print("variables:", variables)
    return __parse(config, variables=variables)

def read_json(file):
    with open(file, "r") as f:
        dic = json.load(f)
    return dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("file", help="")
    parser.add_argument("--test", action="store_true", help="specify src.json and check whether the result matches src.converted.json")
    args = parser.parse_args()

    if args.test:
        tar_config = (args.file).replace(".json", ".converted.json")
        src_config = read_json(args.file)
        tar_config = read_json(tar_config)
        assert tar_config == parse(src_config)
        print("test passed")
    else:
        src_config = read_json(args.file)
        converted = parse(src_config)
        print(converted)
