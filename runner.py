import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from fuzzywuzzy import process
from langdetect import detect
from googletrans import Translator
import random
import numpy as np
import matplotlib.pyplot as plt

# Datasets importing
food = pd.read_csv('./Dataset/food.csv')
df = pd.read_csv("./Dataset/diet.csv")

food.describe()
food.isnull().sum()
food.duplicated().sum()

df.isnull().sum()
df.duplicated().sum()

df = df.drop(columns=['Price'])
df.drop(columns=['Meal_Id'], inplace=True)
df.dropna(subset=['description'], inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Clustering of veg-nonveg
food['VegNovVeg'] = food['VegNovVeg'].apply(lambda x: 1 if x == 'Non-Veg' else 0)

non_veg_keywords = ['Tuna', 'Chicken', 'Salmon', 'Goat', 'Rabbit', 'Pork', 'Bacon', 'Shrimp',
                    'Meatballs', 'Beef', 'Turkey', 'Oyster']

def is_non_veg(food_item):
    for keyword in non_veg_keywords:
        if keyword.lower() in food_item.lower():
            return True
    return False

food['VegNonVeg'] = food['Food_items'].apply(lambda x: 'Non-Veg' if is_non_veg(x) else 'Veg')

food_items = food['Food_items'].tolist()

label_encoder = LabelEncoder()
df['Veg_Non'] = label_encoder.fit_transform(df['Veg_Non'])

X = df[['Veg_Non']]
k = 2
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

cluster_labels = kmeans.labels_
cluster_names = {0: 'veg', 1: 'non-veg'}
df['Cluster'] = [cluster_names[label] for label in cluster_labels]

# Disease
def get_mealfordisease(df, disease_name):
    food_items = df.loc[df['Disease'].str.contains(disease_name), 'Name'].tolist()
    print(f"Meal List for {disease_name}:")
    print('-' * 20)  
    if food_items:
        if len(food_items) <= 10:
            for food_item in food_items:
                print(food_item)
        else:
            random_selection = random.sample(food_items, 10)
            for food_item in random_selection:
                print(food_item)

# Allergy
def translate_description(description):
    translator = Translator()
    translated_text = translator.translate(description, src='auto', dest='en').text
    return translated_text

def get_mealforallergy(df, allergy):
    for index, row in df.iterrows():
        if detect(row['description']) != 'en':
            row['description'] = translate_description(row['description'])

    matching_rows = df[~df['description'].str.contains(allergy, case=False)]

    if not matching_rows.empty:
        food_items = matching_rows['Name'].tolist()
        print(f"Food items where '{allergy}' is not present in the Description:")
        print('-' * 20)
        if food_items:
            if len(food_items) <= 10:
                for food_item in food_items:
                    print(food_item)
            else:
                random_selection = random.sample(food_items, 10)
                for food_item in random_selection:
                    print(food_item)
    else:
        print(f"No food items found where '{allergy}' is not present in the Description.")

# User parameters
def calculate_bmr(age, weight_kg, height_m, gender):
    if gender.lower() == 'female':
        return 655 + (9.6 * weight_kg) + (1.8 * height_m * 100) - (4.7 * age)
    elif gender.lower() == 'male':
        return 66 + (13.7 * weight_kg) + (5 * height_m * 100) - (6.8 * age)
    else:
        raise ValueError("Invalid gender value")

def determine_protein_percentage(veg_nonveg):
    if veg_nonveg.lower() == 'veg':
        return 0.25  
    elif veg_nonveg.lower() == 'nonveg':
        return 0.35  
    else:
        raise ValueError("Invalid vegetarian/non-vegetarian preference")

def determine_water_intake(weight, activity_level):
    activity_levels_ml_per_kg = {
        'sedentary': 30,
        'lightly active': 35,
        'moderately active': 40,
        'very active': 45,
        'extra active': 50
    }
    return weight * activity_levels_ml_per_kg[activity_level.lower()]

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

def display_macros(macros):
    print("\nRecommended nutrition intake:")
    print("TDEE:", macros['TDEE'])
    print("Recommended Protein intake (g):", macros['protein'])
    print("Recommended Fat intake (g):", macros['fat'])
    print("Recommended Carbohydrate intake (g):", macros['carbs'])
    print("Recommended Water intake (ml):", macros['water'])

macros = calculate_macros(weight, height, age, gender, activity_level, veg_nonveg)
display_macros(macros)

def calculate_sums(random_sets):
    sums = {'Calories': 0, 'Proteins': 0, 'Fats': 0, 'Carbohydrates': 0}
    for set_values in random_sets:
        for i, nutrient in enumerate(['Calories', 'Proteins', 'Fats', 'Carbohydrates']):
            sums[nutrient] += set_values[i]
    return sums

def generate_nutrient_sets(calories_total, proteins_total, fats_total, carbohydrates_total, num_sets=10):
    calories_target = calories_total // num_sets
    proteins_target = proteins_total // num_sets
    fats_target = fats_total // num_sets
    carbohydrates_target = carbohydrates_total // num_sets
    sets = []
    counter = 0
    
    while len(sets) < num_sets:
        calories = random.randint(calories_target - 250, calories_target + 250)
        proteins = random.randint(proteins_target - 15, proteins_target + 15)
        fats = random.randint(fats_target - 10, fats_target + 10)
        carbohydrates = random.randint(carbohydrates_target - 25, carbohydrates_target + 25)
        
        if not any(all(value == nutrient for value in nutrient_set) for nutrient_set in sets for nutrient in (calories, proteins, fats, carbohydrates)):
            sets.append((calories, proteins, fats, carbohydrates))
            counter += 1
    
    return sets

def generate_sets_from_nutrient_recommendations(nutrient_recommendations):
    sets = []
    
    for recommendation in nutrient_recommendations:
        calories, proteins, fats, carbohydrates = recommendation
        sets.append({
            'calories': calories,
            'proteins': proteins,
            'fats': fats,
            'carbohydrates': carbohydrates
        })
    
    return sets

# Prediction of food from dataset 1
def food_food_subset(veg_nonveg, food):
    if veg_nonveg.lower() == 'veg':
        return food[food['VegNonVeg'] == 'Veg']
    elif veg_nonveg.lower() == 'non-veg':
        return food  
    else:
        raise ValueError("Invalid input.")

def predict_food_names(sets, knn, food):
    predicted_food_names = set()
    for set_values in sets:
        X_input = np.array(set_values).reshape(1, -1)
        predicted_index = knn.predict(X_input)
        predicted_food_name = food.loc[predicted_index[0], 'Food_items']
        if predicted_food_name not in predicted_food_names:
            predicted_food_names.add(predicted_food_name)
    
    return predicted_food_names

food_subset = food_food_subset(veg_nonveg, food)
generated_sets = generate_nutrient_sets(macros['TDEE'], macros['protein'], macros['fat'], macros['carbs'])

X = food_subset[['Calories', 'Proteins', 'Fats', 'Carbohydrates']]  
y = food_subset.index  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 1  
knn_veg = KNeighborsClassifier(n_neighbors=k)
knn_veg.fit(X_train, y_train)

predicted_food_names = predict_food_names(generated_sets, knn_veg, food_subset)

# Dishes from 2nd Dataset
def translate_description(description):
    if detect(description) != 'en':
        translated = Translator().translate(description, src='auto', dest='en')
        return translated.text
    else:
        return description

def df_food_subset(veg_nonveg, df):
    if veg_nonveg.lower() == 'veg':
        return df[df['Cluster'] == 'veg']  
    elif veg_nonveg.lower() == 'non-veg':
        return df  
    else:
        raise ValueError("Invalid input.")

def search_food_names(df_filtered, search_words):
    printed_food_names = set()
    printed_search_words = set()
    
    for _, row in df_filtered.iterrows():
        translated_description = translate_description(row['description'])
        translated_words = translated_description.split()
        
        for word in search_words:
            if word.lower() in translated_words:
                if row['Name'] not in printed_food_names:
                    printed_food_names.add(row['Name'])
                break
        else:
            if word not in printed_search_words:
                printed_search_words.add(word)

    random_selection = random.sample(printed_search_words.union(printed_food_names), min(10, len(printed_search_words) + len(printed_food_names)))
    for item in random_selection:
        print(item)
        
df_filtered = df_food_subset(veg_nonveg, df)
search_words = ' '.join(predicted_food_names).lower().split()
search_food_names(df_filtered, search_words)

# Graph of ingredient or food listed above
def create_pie_chart(food, predicted_food_names):
    for food_name in predicted_food_names:
        match = process.extractOne(food_name, food['Food_items'])
        if match[1] >= 80:  
            food_item = match[0]  
            item_row = food[food['Food_items'] == food_item]
            
            labels = ['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium','Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']
            sizes = item_row[labels].values[0]
            
            colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen',
                    'pink', 'lightblue', 'orange', 'cyan', 'magenta', 'red']
            
            fig, ax = plt.subplots()
            wedges, _, _ = ax.pie(sizes, colors=colors, autopct='', startangle=140, textprops=dict(color="black"))
            
            ax.legend(wedges, [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes)], title="Nutrients", loc="center left", bbox_to_anchor=(1, 0.5))
            
            veg_nonveg = item_row['VegNovVeg'].values[0]
            if veg_nonveg == 0:
                veg_label = 'Vegetarian'
            else:
                veg_label = 'Non-Vegetarian'
            plt.text(1.5, 1.0, f'{veg_label}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
            
            meal_type = ''
            if item_row['Breakfast'].values[0] == 1:
                meal_type += 'Breakfast'
            if item_row['Lunch'].values[0] == 1:
                if meal_type:
                    meal_type += ', '
                meal_type += 'Lunch'
            if item_row['Dinner'].values[0] == 1:
                if meal_type:
                    meal_type += ', '
                meal_type += 'Dinner'
            plt.text(1.5, 0.9, f'{meal_type}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='lightskyblue', alpha=0.5))
            
            ax.axis('equal')  
            plt.title(f'Nutrient Distribution of {food_item}')
            plt.show()
        else:
            print("Food item not found or not similar enough in the dataset.")   

create_pie_chart(food, predicted_food_names)
