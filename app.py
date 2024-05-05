from flask import Flask, render_template, request
import random
from runner import food, df, knn_model_veg_food, knn_model_food, calculate_macros, food_food_subset, generate_nutrient_sets, predict_food_names, df_food_subset, search_food_names_with_allergy, search_food_names_without_allergy, predicted_food_names_with_allergies, create_pie_chart, get_description

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        # Access form data
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = round(float(request.form['weight']))
        height = float(request.form['height'])
        veg_nonveg = request.form['veg_or_nonveg']
        activity_level = request.form['activity_level']
        allergies = request.form['foodtype'].lower()  # Assuming ID for allergies is 'foodtype'
        
        # Calculate macros based on form data
        macros_dict = calculate_macros(weight, height, age, gender, activity_level, veg_nonveg)
        
        # Handle allergies
        allergies = [allergy.strip().lower() for allergy in allergies.split(',') if allergy.strip()]
        
        # Load models and datasets, perform prediction, and search
        food_subset = food_food_subset(veg_nonveg, food)
        generated_sets = generate_nutrient_sets(macros_dict['TDEE'], macros_dict['protein'], macros_dict['fat'], macros_dict['carbs'])
        
        knn_model = knn_model_veg_food if veg_nonveg.lower() == 'veg' else knn_model_food
        
        predicted_food_names = predict_food_names(generated_sets, knn_model, food_subset)
        
        # Filter food based on allergies
        if allergies is not None:
            predicted_food = predicted_food_names_with_allergies(allergies, predicted_food_names)
        else:
            predicted_food = predicted_food_names
        
        item_pie_chart = create_pie_chart(food, predicted_food)
        
        item_with_pie = zip(predicted_food, item_pie_chart)
        
        df_filtered = df_food_subset(veg_nonveg, df)
        search_words = ' '.join(predicted_food).lower().split()
        
        if allergies is not None:
            printed_food_names, printed_search_words = search_food_names_with_allergy(df_filtered,predicted_food, search_words, allergies)
        else:
            printed_food_names, printed_search_words = search_food_names_without_allergy(df_filtered,predicted_food, search_words)
        
        random_selection = random.sample(list(printed_food_names.union(printed_search_words)), min(10, len(printed_food_names.union(printed_search_words))))
        
        description = get_description(df, random_selection)
        
        recommend_with_desp = zip(random_selection, description)
        
        return render_template('recommend.html', recommendations=random_selection, macros=macros_dict, item=predicted_food, desp=recommend_with_desp, pie=item_with_pie)
    
    return render_template('reco.html')

if __name__ == '__main__':
    app.run(debug=True)
