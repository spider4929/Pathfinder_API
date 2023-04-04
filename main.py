from flask import Flask, request
from pathfinder import pathfinder, text_to_speech_safest, text_to_speech_fastest

app = Flask(__name__)


@app.route("/")
def hello():
    return '<h1>PathFinder API running</h1>'


@app.route("/route/", methods=["POST"])
def find_path():
    args = request.get_json()
    orig = args['sourceCoords']
    dest = args['destCoords']
    prefs = args['preferences']

    y_orig = float(orig[1])
    x_orig = float(orig[0])

    y_dest = float(dest[1])
    x_dest = float(dest[0])

    user_pref = {}

    for item in prefs:
        if item['value'] == 0:
            user_pref[item['name']] = 0
        else:
            user_pref[item['name']] = item['value']/5

    try:
        if args['adjust']:
            adjust = False
    except:
        adjust = True

    route = pathfinder([y_orig, x_orig], [y_dest, x_dest], adjust, user_pref)

    return route


@app.route("/steps_with_coords_safest/", methods=["POST"])
def steps_with_coords_safest():
    args = request.get_json()
    orig = args['sourceCoords']
    dest = args['destCoords']
    prefs = args['preferences']

    y_orig = float(orig[1])
    x_orig = float(orig[0])

    y_dest = float(dest[1])
    x_dest = float(dest[0])

    user_pref = {}

    for item in prefs:
        if item['value'] == 0:
            user_pref[item['name']] = 0
        else:
            user_pref[item['name']] = item['value']/5

    try:
        if args['adjust']:
            adjust = False
    except:
        adjust = True

    route = text_to_speech_safest(
        [y_orig, x_orig], [y_dest, x_dest], adjust, user_pref)

    return route


@app.route("/steps_with_coords_fastest/", methods=["POST"])
def steps_with_coords_fastest():
    args = request.get_json()
    orig = args['sourceCoords']
    dest = args['destCoords']

    y_orig = float(orig[1])
    x_orig = float(orig[0])

    y_dest = float(dest[1])
    x_dest = float(dest[0])

    route = text_to_speech_fastest(
        [y_orig, x_orig], [y_dest, x_dest])

    return route

if __name__ == '__main__':
  app.run(port=5000)
