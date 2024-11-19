import json
import os

JSON_INDENT = 2
N_METADATA_SPLITS = 2
METADATA_BASE_PATH = "data/bronze/metadata/dish_metadata_cafe{}.csv"
SILVER_DATA_PATH = "data/silver/metadata"
METADATA_FIELDS = [
    {"key": "dish_id", "type_converter": str},
    {"key": "total_calories", "type_converter": float},
    {"key": "total_mass", "type_converter": float},
    {"key": "total_fat", "type_converter": float},
    {"key": "total_carb", "type_converter": float},
    {"key": "total_protein", "type_converter": float}
]

def main():
    metadata = get_metadata()
    dishes_info = get_dishes_info(metadata)

    if not os.path.isdir(SILVER_DATA_PATH):
        os.mkdir(SILVER_DATA_PATH)

    for dish_info in dishes_info:
        save_dish_info(dish_info)


def get_metadata() -> list:
    metadata = []
    for i in range(N_METADATA_SPLITS):
        with open(METADATA_BASE_PATH.format(i+1)) as f:
            lines = f.readlines()
            current_split_metadata = [line.removesuffix("\n") for line in lines]
            metadata.extend(current_split_metadata)
    
    return metadata


def get_dishes_info(metadata: list) -> list:
    dishes_info = []
    for sample in metadata:
        sample_data = sample.split(",")
        dish_info = {}
        for index, field_info in enumerate(METADATA_FIELDS):
            key = field_info["key"]
            type_converter = field_info["type_converter"]

            dish_info[key] = type_converter(sample_data[index])

        dishes_info.append(dish_info)

    return dishes_info


def save_dish_info(dish_info):
    dish_metadata_filename = f"{SILVER_DATA_PATH}/{dish_info['dish_id']}.json"
    with open(dish_metadata_filename, "w") as dish_metadata_file:
        json.dump(dish_info, dish_metadata_file, indent=JSON_INDENT)


if __name__ == "__main__":
    main()