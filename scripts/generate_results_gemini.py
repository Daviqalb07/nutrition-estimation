import os
import json
import time
from functools import reduce
import google.generativeai as genai
import typing_extensions as typing
from dotenv import load_dotenv

load_dotenv()

IMAGES_BASE_PATH="data/bronze/imagery/realsense_overhead"
TEST_IDS_PATH="data/bronze/dish_ids/splits/rgb_test_ids.txt"
RESULTS_PATH="results/gemini"
INDENT_SIZE=2


def main():
    with open(TEST_IDS_PATH, "r") as file:
        test_ids = [line.removesuffix("\n") for line in file.readlines()]

    dishes_ids_with_images = set(os.listdir(IMAGES_BASE_PATH))
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

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
            except Exception as e:
                print(e)
    


def upload_meal_image_from_id(mealId: str):
    image_path = f"{IMAGES_BASE_PATH}/{mealId}/rgb.png"
    file = genai.upload_file(path=image_path)
    
    return file


class FoodItem(typing.TypedDict):
    name: str
    portion: str
    calories: int
    carbohydrates: int


def estimate_nutrition_info(image_file, model):
    prompt = """
        Você é um especialista em nutrição. Analise a imagem dessa refeição, reconheça os alimentos
        e estime as porções de cada um. Em seguida, com base nas porções estimadas, calcule a quantidade
        de calorias e carboidratos (em gramas) para cada alimento.
    """

    start_time = time.time()
    response = model.generate_content(
        [image_file, prompt],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", 
            response_schema=list[FoodItem]
        )
    )
    elapsed_time = time.time() - start_time
    return json.loads(response.text), elapsed_time


def save_result(data):
    dish_id = data["dishId"]
    result_path = f"{RESULTS_PATH}/{dish_id}.json"
    with open(result_path, "w") as result_file:
        json.dump(data, result_file, indent=INDENT_SIZE)


if __name__ == "__main__":
    main()