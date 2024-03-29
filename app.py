from flask import Flask, render_template, request
#import runner

app = Flask(__name__)

def calculate_bmr(age, weight_kg, height_m, gender):
    if gender.lower() == 'female':
        return 655 + (9.6 * weight_kg) + (1.8 * height_m * 100) - (4.7 * age)
    elif gender.lower() == 'male':
        return 66 + (13.7 * weight_kg) + (5 * height_m * 100) - (6.8 * age)
    else:
        raise ValueError("Invalid gender value")

def calculate_macros(weight, height, age, gender, activity_level, veg_nonveg):
    ree = calculate_bmr(age, weight, height, gender)

    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }

    tdee = ree * activity_factors[activity_level.lower()]

    protein_percent = 0.3
    fat_percent = 0.25
    carb_percent = 1 - protein_percent - fat_percent

    # Adjust protein percentage based on diet type (veg or non-veg)
    protein_percent = determine_protein_percentage(veg_nonveg)

    protein_calories = protein_percent * tdee
    fat_calories = fat_percent * tdee
    carb_calories = carb_percent * tdee

    protein_grams = protein_calories / 4  
    fat_grams = fat_calories / 9  
    carb_grams = carb_calories / 4  

    water_intake_ml = determine_water_intake(weight, activity_level)

    calorie_intake = tdee

    return {
        'TDEE': tdee,
        'protein': protein_grams,
        'fat': fat_grams,
        'carbs': carb_grams,
        'water': water_intake_ml
    }

def determine_protein_percentage(veg_nonveg):
    # Adjust protein percentage based on diet type (veg or non-veg)
    if veg_nonveg.lower() == 'veg':
        return 0.25  # Example value for vegetarians
    elif veg_nonveg.lower() == 'nonveg':
        return 0.35  # Example value for non-vegetarians
    else:
        raise ValueError("Invalid diet type")

def determine_water_intake(weight, activity_level):
    # Dummy function to determine water intake based on weight and activity level
    return weight * 30  # Example value

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
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        veg_nonveg = request.form['veg_or_nonveg']
        activity_level = request.form['activity_level']
        
        # Calculate macros based on form data
        macros_dict = calculate_macros(weight, height, age, gender, activity_level, veg_nonveg)
        
        # Print the calculated macros
        print("Macros:", macros_dict)

        # Dummy example for demonstration
        recommendations = ["Recommendation 1", "Recommendation 2", "Recommendation 3"]

        return render_template('recommend.html', recommendations=recommendations, macros=macros_dict)

    return render_template('reco.html')

if __name__ == '__main__':
    app.run(debug=True)
