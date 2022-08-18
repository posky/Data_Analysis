import os
import pickle

from flask import Flask, render_template, request
import numpy as np

model = pickle.load(
    open(
        os.path.join(os.path.dirname(__file__), 'model/mobile_price.pkl'), 'rb'
    )
)
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def start():
    data = []
    data.append(float(request.form['battery_power']))
    data.append(int(request.form['blue']))
    data.append(float(request.form['clock_speed']))
    data.append(int(request.form['daul_sim']))
    data.append(int(request.form['fc']))
    data.append(int(request.form['four_g']))
    data.append(int(request.form['int_memory']))
    data.append(float(request.form['m_dep']))
    data.append(int(request.form['mobile_wt']))
    data.append(int(request.form['n_cores']))
    data.append(int(request.form['pc']))
    data.append(int(request.form['px_height']))
    data.append(int(request.form['px_width']))
    data.append(int(request.form['ram']))
    data.append(int(request.form['sc_h']))
    data.append(int(request.form['sc_w']))
    data.append(int(request.form['talk_time']))
    data.append(int(request.form['three_g']))
    data.append(int(request.form['touch_screen']))
    data.append(int(request.form['wifi']))

    arr = np.array([data])
    pred = model.predict(arr)
    print('start predict', pred)
    return render_template('predict.html', data=pred)


if __name__ == '__main__':
    app.run(debug=True)
