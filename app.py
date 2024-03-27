from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reco', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        # Access form data
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        veg_or_nonveg = request.form['veg_or_nonveg']
        disease = request.form['disease']
        food_type = request.form['foodtype']

        # Perform recommendation logic here
        # You can use the input data to generate recommendations

        # Dummy example for demonstration
        recommendations = ["Recommendation 1", "Recommendation 2", "Recommendation 3"]

        return render_template('recommendations.html', recommendations=recommendations)

    return render_template('reco.html')

if __name__ == '__main__':
    app.run(debug=True)
