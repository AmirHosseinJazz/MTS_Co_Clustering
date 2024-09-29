import os
import subprocess
import pandas as pd


def evaluation():
    # print(os.listdir("../TVAE/saved_models"))
    models = os.listdir("../TVAE/saved_models")
    for k in models[:]:
        print(k)
        cmd = [
            "python",
            "eval.py",
            "--method",
            "all",
            "--model_name",
            k,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)


def parse_value(value):
    """
    Parse the value from string format to appropriate Python data types
    like int, float, bool, list, or tuple.
    """
    try:
        # Convert to boolean if applicable
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False

        # Convert to int, float if possible
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                pass

        # Convert to list or tuple if applicable
        if value.startswith("[") and value.endswith("]"):
            return eval(value)  # Evaluates lists e.g., [50, 100, 200]
        elif value.startswith("(") and value.endswith(")"):
            return eval(value)  # Evaluates tuples e.g., (4, 8)
    except Exception:
        pass

    # Return as string if no other type matches
    return value


def txt_to_dict(file_path):
    """
    Reads a key-value formatted txt file and converts it to a dictionary.
    """
    data_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            # Split the line at the first occurrence of ':'
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Parse the value to correct type
            data_dict[key] = parse_value(value)

    return data_dict


def kl_divergence():
    # print(os.listdir("../TVAE/saved_models"))
    models = os.listdir("../TVAE/saved_models")
    measures = []
    for k in models[:]:
        single = pd.read_csv("./marginal_dist/" + k + "/ks_stats.csv")
        mean = single["ks_stat"].mean()
        mean_p_value = single["p_value"].mean()
        # read txt file convert to dict
        config = txt_to_dict("../TVAE/saved_models/" + k + "/config.txt")
        kl = txt_to_dict("./marginal_dist/" + k + "/kl_divergence.txt")
        mean_kl = pd.DataFrame([kl]).T.mean()[0]
        # print(mean_kl)
        measures.append(
            {
                "model": k,
                "ks": mean,
                "latent_dim": config["latent_dim"],
                "kl": mean_kl,
                "p_value": mean_p_value,
            }
        )

    pd.DataFrame(measures).to_csv("kl_divergence.csv", index=False)


if __name__ == "__main__":
    # evaluation()
    kl_divergence()
