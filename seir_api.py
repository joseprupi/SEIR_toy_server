from flask import Flask,request, jsonify

from SEIR import SEIR

app = Flask(__name__)

seir = SEIR()

@app.route('/api/v1.0/get_SEIR', methods=['POST'])
def getseir():
    content = request.json
    seir.set_params(float(content['T_inc']), float(content['T_inf']), float(content['R_0']))
    sol = seir.solve()
    return jsonify({'result': sol.y})

if __name__ == '__main__':
    app.run()
