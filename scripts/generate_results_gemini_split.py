import os
import json
import time
from functools import reduce
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import typing_extensions as typing
from dotenv import load_dotenv

load_dotenv()

IMAGES_BASE_PATH="data/bronze/imagery/realsense_overhead"
RESOURCES_PATH="resources/"
TEST_IDS_PATH="data/bronze/dish_ids/splits/rgb_test_ids.txt"
RESULTS_PATH="results/gemini_split"
INDENT_SIZE=2

def main():
    with open(TEST_IDS_PATH, "r") as file:
        test_ids = [line.removesuffix("\n") for line in file.readlines()]

    dishes_ids_with_images = set(os.listdir(IMAGES_BASE_PATH))
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = create_model()

    for test_id in test_ids:
        if test_id in dishes_ids_with_images:
            try:
                image_file = upload_meal_image_from_id(test_id)
                nutrition_info, elapsed_time = estimate_nutrition_info(image_file, model)

                print(f"{test_id} processed ({elapsed_time} s)")
                image_file.delete()
                
                result_data = {
                    "dishId": test_id,
                    "nutritionData": nutrition_info,
                    "totalCalories": reduce(lambda sum, data: sum + data["calories"], nutrition_info, 0),
                    "totalCarbohydrates": reduce(lambda sum, data: sum + data["carbohydrates"], nutrition_info, 0),
                    "elapsedTime": elapsed_time
                }

                save_result(result_data)
                time.sleep(2)
            except Exception as e:
                print(f"Failed to process {test_id}: {e}")


def create_model():
    return genai.GenerativeModel(model_name="gemini-1.5-flash")


def upload_meal_image_from_id(mealId: str):
    image_path = f"{IMAGES_BASE_PATH}/{mealId}/rgb.png"
    file = genai.upload_file(path=image_path)
    
    return file


def estimate_nutrition_info(image_file, model: genai.GenerativeModel):
    start_time = time.time()
    food_items = get_food_items(image_file, model)
    nutrition_info = get_nutrition_info(food_items, model)
    elapsed_time = time.time() - start_time

    return nutrition_info, elapsed_time


def get_food_items(image_file, model: genai.GenerativeModel):
    prompt = """
        Analise a imagem da refeição, reconheça os alimentos e estime as porções de cada um.
        Retorne para cada alimento o seu nome e a porção, em português.
    """

    response = model.generate_content(
        [image_file, prompt],
        generation_config=genai.GenerationConfig(            
            response_schema=content.Schema(
                type = content.Type.OBJECT,
                properties = {
                    "foodItems": content.Schema(
                        type = content.Type.ARRAY,
                        items = content.Schema(
                            type = content.Type.OBJECT,
                            properties = {
                                "name": content.Schema(
                                    type = content.Type.STRING,
                                    description = "O nome do alimento (em português)"
                                ),
                                "portion": content.Schema(
                                    type = content.Type.STRING,
                                    description = "A porção estimada (em português)"
                                )
                            },
                        ),
                    ),
                },
            ),
            response_mime_type="application/json"
        )
    )

    return response.text


def get_nutrition_info(food_items: str, model: genai.GenerativeModel):
    prompt = f"""
    {food_items}
    Cada item nesse array JSON representa um alimento e a porção dele numa refeição.
    Para cada alimento, calcule a quantidade de calorias (em kcal) e carboidratos (em gramas) de acordo com a porção dada.
    """

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_schema=content.Schema(
                type = content.Type.OBJECT,
                properties = {
                    "nutritionData": content.Schema(
                        type = content.Type.ARRAY,
                        items = content.Schema(
                            type = content.Type.OBJECT,
                            properties = {
                                "name": content.Schema(
                                    type = content.Type.STRING,
                                    description = "O nome do alimento (em português)"
                                ),
                                "portion": content.Schema(
                                    type = content.Type.STRING,
                                    description = "A porção estimada (em português)"
                                ),
                                "calories": content.Schema(
                                    type = content.Type.INTEGER,
                                    description = "A quantidade de calorias (em kcal)"
                                ),
                                "carbohydrates": content.Schema(
                                    type = content.Type.INTEGER,
                                    description = "A quantidade de carboidratos (em gramas)"
                                ),
                            },
                        ),
                    ),
                },
            ),
            response_mime_type="application/json"
        )
    )

    parsed_response = json.loads(response.text)
    return parsed_response["nutritionData"]

def save_result(data):
    dish_id = data["dishId"]
    result_path = f"{RESULTS_PATH}/{dish_id}.json"
    with open(result_path, "w") as result_file:
        json.dump(data, result_file, indent=INDENT_SIZE)


if __name__ == "__main__":
    main()