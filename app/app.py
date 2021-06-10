from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


def make_list_to_df(list):
    df = pd.DataFrame(columns=['YearsCodePro', 'Age',
                               'Country', 'DevType', 'EdLevel'])
    df.loc[len(df)] = list
    return df


def load_models():
    file_name = "linear_reg_model.sav"
    linear_reg_model_reloaded = pickle.load(open(file_name, 'rb'))

    return linear_reg_model_reloaded


# load the model from disk #
#filename = 'linear_reg_model.sav'
#linear_reg_model_reloaded = pickle.load(open(filename, 'rb'))


def salary_prediction(model, new_data):
    salary = model.predict(new_data)
    return salary


@app.route("/", methods=["GET", "POST"])
def hello_world():
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template("index.html")
    else:
        yearscode = request.form["yearscode"]
        age = request.form["age"]
        country = request.form["country"]
        devtype = request.form["devtype"]
        education = request.form["education"]
        new_entry = [yearscode, age, country, devtype, education]
        #new_entry = ['0', '25', 'United States','Data-scientist', 'Bachelors-degree']

        new_entry_df = make_list_to_df(new_entry)

        # import model #
        model = load_models()
        model_result = salary_prediction(model, new_entry_df)
        model_result_str = str(np.rint(model_result[0]))

        return render_template("index.html", FinalOutput=model_result_str)

        # return render_template("index.html", outputyearscode=yearscode, outputage=age, outputcountry=country, outputdevtype=devtype, outputeducation=education)


# if __name__ == "__main__":
#     app.run(debug=True)
