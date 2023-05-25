from flask import Flask, request
from pathfinder import pathfinder, get_nearest_edge
from datetime import datetime

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

    route = pathfinder([y_orig, x_orig], [y_dest, x_dest], user_pref)

    now = datetime.now()

    print(f'/route/ Success - {now}')

    return route

@app.route("/route/get_nearest_edge/", methods=["POST"])
def nearest_edge():
    args = request.get_json()
    coords = args['coords']

    y_coord = float(coords[1])
    x_coord = float(coords[0])

    edges = get_nearest_edge(y_coord, x_coord)

    now = datetime.now()

    print(f'route/get_nearest_edge/ Success - {now}')

    return edges

if __name__ == '__main__':
    app.run(port=5000)
