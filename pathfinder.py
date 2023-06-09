import osmnx
import networkx as nx
import polyline
from heapq import heappop, heappush
from itertools import count
from datetime import datetime
import requests
import pymongo
from haversine import haversine

#### FUNCTIONS ####


def api_profile(weather, profile):
    """Adjusts the current profile according to current weather and time conditions."""
    new_profile = profile
    case = [True, True]

    now = datetime.now().hour + 8
    if now >= 24:
        now = now - 24

    # Removes 'flood_hazard' if weather is clear
    if weather not in [201, 202, 211, 212, 221, 501, 502, 503, 504, 511, 521, 522, 531]:
        new_profile.pop("not_flood_hazard")
        case[0] = False

    # Removes 'lighting' if time is day
    if now not in [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]:
        new_profile.pop("lighting")
        case[1] = False

    return new_profile, case


def adjust_weight(length, row, profile):
    """Returns the weight value for the given 'edge/row' parameters based on 'length' and 'profile'."""
    weight = length
    modifier = 1
    for safety_factor, user_preference in profile.items():
        # if row['closed'] == '1':
        #     modifier = 999999
        #     break
        if row[safety_factor] == '0':
            modifier += user_preference
    return weight * modifier

#### PATH FINDING ####


def getCoordinates(route, nodes, origin, destination):
    """Returns coordinates of route as an array of tuples."""
    final_coords = []

    final_coords.append({
        'latitude': origin['y'],
        'longitude': origin['x']
    })

    for id in route:
        coord = {
            'latitude': nodes.filter(items=[id], axis=0).y.item(),
            'longitude': nodes.filter(items=[id], axis=0).x.item()
        }
        final_coords.append(coord)

    final_coords.append({
        'latitude': destination['y'],
        'longitude': destination['x']
    })

    return final_coords


def getPolyline(route, nodes):
    """Returns encoded polyline using the polyline library, requires input as an array of tuples."""
    coordinates = getCoordinates(route, nodes)
    return polyline.encode(coordinates)


def getRouteLength(route, graph):
    """Function to return the total length of the route in meters."""
    route_length = osmnx.utils_graph.get_route_edge_attributes(
        graph, route, attribute='length')
    return round(sum(route_length))


def getBearingString(degrees, name):
    """Convert bearing value to readable instruction."""
    instruction = None
    if degrees < 45:
        instruction = 'Head North '
    elif degrees < 135:
        instruction = 'Head East '
    elif degrees < 225:
        instruction = 'Head South '
    elif degrees < 315:
        instruction = 'Head West '
    else:
        instruction = 'Head North '
    if isinstance(name, list):
        name = ' '.join(name)
    if name == '':
        return instruction
    return instruction + 'along ' + name


def getManeuever(heading, true_bearing):
    """Get maneuever type based on bearing relative_bearing."""
    relative_bearing = abs(true_bearing - heading)
    if relative_bearing <= 45 or relative_bearing >= 315:
        return 'straight'
    return 'turn'


def getTurnDirection(heading, true_bearing, name):
    """Translate turn direction in bearings to string."""
    relative_bearing = true_bearing - heading
    if relative_bearing < 0:
        relative_bearing += 360
    instruction = None
    if relative_bearing <= 45 or relative_bearing >= 315:
        instruction = 'Continue Straight '
    if relative_bearing >= 45 and relative_bearing < 180:
        instruction = 'Turn Right '
    if relative_bearing > 180 and relative_bearing <= 315:
        instruction = 'Turn Left '
    if isinstance(name, list):
        name = ' '.join(name)
    if name == '':
        return instruction
    try:
        return instruction + 'onto ' + name
    except:
        return instruction


def getRouteDirections(route, nodes, graph, safety_factors):
    """Generates the step-by-step instructions in a given 'route'.
    'nodes' are used to evaluate coordinates per step
    'graph' are used to generate the bearing values to calculate for turn directions
    'safety_factors' is the profile mask used to collect edge attributes of the 'route' in the 'graph'
    """
    # generate edge bearings for graph
    bearings_graph = osmnx.bearing.add_edge_bearings(graph, precision=1)
    # Generate a dictionary of relevant keys of the route for directions
    steps = osmnx.utils_graph.get_route_edge_attributes(bearings_graph, route)

    direction = []
    safety_coverage_direction = []
    instruction = None
    maneuever = None  # depart - first step, turn - for any step in between, arrive - last step
    distance = None
    name = None
    before_maneuever = None
    before_name = None
    footway = None
    present_factors = []
    bearing_before = 0
    bearing_after = 0
    count = 0
    # Start parsing
    for count, step in enumerate(steps):
        before_factors = present_factors
        present_factors = []
        before_name = name
        before_maneuever = maneuever
        name = step.get("name", "")
        footway = step.get("footway")
        distance = round(step.get("length"))

        for factor in safety_factors:
            if step.get(factor) == '1':
                present_factors.append(factor)

        # If the step is the first step
        if steps[0] == step:
            bearing_before = 0
            bearing_after = step.get("bearing")
            maneuever = 'depart'
            instruction = getBearingString(bearing_after, name)
            direction.append({'maneuever': maneuever,
                              'footway': footway,
                              'instruction': instruction,
                              'name': name,
                              #   'bearing_before': bearing_before,
                              #   'bearing_after': bearing_after,
                              'distance': distance,
                              'factors_present': present_factors,
                              'coordinates': [
                                  nodes.filter(
                                      items=[route[count]], axis=0).y.item(),
                                  nodes.filter(
                                      items=[route[count]], axis=0).x.item()
                              ]})

            safety_coverage_direction.append({
                'distance': distance,
                'factors_present': present_factors})
            continue

        # If the step is any steps in between the first and last step
        if steps[0] != step and steps[-1] != step:
            bearing_before = bearing_after
            bearing_after = step.get("bearing")
            maneuever = getManeuever(bearing_before, bearing_after)

            if footway == 'crossing':
                instruction = 'Cross the street'
            elif footway == 'footbridge':
                instruction = 'Cross the street via the footbridge'
            else:
                instruction = getTurnDirection(
                    bearing_before, bearing_after, name)

            safety_coverage_direction.append({'distance': distance,
                                              'factors_present': present_factors})

            if before_name == name and before_maneuever == maneuever and before_factors == present_factors:
                direction[-1]["distance"] += distance
                for factor in present_factors:
                    if factor not in direction[-1]["factors_present"]:
                        direction[-1]["factors_present"].append(factor)
                continue

            direction.append({'maneuever': maneuever,
                              'footway': footway,
                              'instruction': instruction,
                              'name': name,
                              #   'bearing_before': bearing_before,
                              #   'bearing_after': bearing_after,
                              'distance': distance,
                              'factors_present': present_factors,
                              'coordinates': [
                                  nodes.filter(
                                      items=[route[count]], axis=0).y.item(),
                                  nodes.filter(
                                      items=[route[count]], axis=0).x.item()
                              ]})
            continue

        # If the step is the last step
        if steps[-1] == step:
            bearing_before = bearing_after
            bearing_after = step.get("bearing")
            maneuever = 'arrive'
            instruction = getTurnDirection(
                bearing_before, bearing_after, name) + " and arrive at destination"
            safety_coverage_direction.append({
                'distance': distance,
                'factors_present': present_factors,
            })
            direction.append({'maneuever': maneuever,
                              'footway': footway,
                              'instruction': instruction,
                              'name': name,
                              #   'bearing_before': bearing_before,
                              #   'bearing_after': bearing_after,
                              'distance': distance,
                              'factors_present': present_factors,
                              'coordinates': [
                                  nodes.filter(
                                      items=[route[count]], axis=0).y.item(),
                                  nodes.filter(
                                      items=[route[count]], axis=0).x.item()
                              ]})

    return direction, safety_coverage_direction


def getSafetyFactorCoverage(steps, length, safety_factors, profile):
    """Evaluates the safety factor coverage for each safety factor for the route.
    The calculation is based on the coverage of the safety factor (in m) divided by the total distance (in m) of the route.
    """
    factor_coverage = {
        'not_flood_hazard': 0,
        'pwd_friendly': 0,
        'cctv': 0,
        'landmark': 0,
        'lighting': 0,
        'not_major_road': 0
    }
    temp = 0

    for step in steps:
        for factor in safety_factors:
            if factor in step['factors_present']:
                factor_coverage[factor] += step['distance']
            else:
                pass

    for factor in safety_factors:
        factor_coverage[factor] = round((factor_coverage[factor]/length)*100)

    for item in profile.keys():
        if item in factor_coverage.keys():
            temp += factor_coverage[item] * profile[item]

    if sum(profile.values()) == 0:
        temp = sum(factor_coverage.values())/6
    else:
        temp = temp/sum(profile.values())

    factor_coverage['average'] = round(temp)

    return factor_coverage


def report_update_graph(edges, origin, destination, optional_id=None):
    client = pymongo.MongoClient(
        "mongodb+srv://team-5-design-project:WJh3Yqe7bLgGwTEr@pathfinder.9orhde9.mongodb.net/?retryWrites=true&w=majority")

    db = client["test"]

    collection = db["reports"]
    db_report = collection.find()
    origin = (origin['y'], origin['x'])
    dest = (destination['y'], destination['x'])
    threshold = haversine(origin, dest)

    reports = []

    for report in db_report:
        coords = (report['coordinates']['latitude'], report['coordinates']['longitude'])
        if haversine(origin, coords) >= threshold:
            pass
        else:
            reports.append(report)

    client.close()

    report_categories = {
        'landmark': 'landmark',
        'lighting': 'lighting',
        'pwd': 'pwd_friendly',
        'cctv': 'cctv',
        'flood': 'not_flood_hazard'
    }

    if not reports:
        pass
    else:
        for report in reports:
            if optional_id and optional_id == report['_id']:
                nearest_edge = eval(report['edges'])
                edges.loc[(nearest_edge[0], nearest_edge[1]), 'road_closed'] = '1'
                edges.loc[(nearest_edge[1], nearest_edge[0]), 'road_closed'] = '1'
            if 'closure' in report['category']:
                if report['counter'] >= 5:
                    nearest_edge = eval(report['edges'])
                    edges.loc[(nearest_edge[0], nearest_edge[1]), 'road_closed'] = '1'
                    edges.loc[(nearest_edge[1], nearest_edge[0]), 'road_closed'] = '1'
                else:
                    pass
            elif 'not' in report['category']:
                nearest_edge = eval(report['edges'])
                category = report_categories[report['category'][4:]]
                edges.loc[(nearest_edge[0], nearest_edge[1]), category] = '0'
                edges.loc[(nearest_edge[1], nearest_edge[0]), category] = '0'
            else:
                nearest_edge = eval(report['edges'])
                category = report_categories[report['category']]
                edges.loc[(nearest_edge[0], nearest_edge[1]), category] = '1'
                edges.loc[(nearest_edge[1], nearest_edge[0]), category] = '1'

    return edges

def get_nearest_edge(y_coord, x_coord):

    # graph = osmnx.graph_from_xml('C:\\Users\\kjqb4\\Documents\\GitHub Projects\\design-project\\Pathfinder_API\\map_complete.osm', simplify=False)
    graph = osmnx.graph_from_xml('map_complete.osm', simplify=False)
    
    nearest_edges = osmnx.nearest_edges(
            graph, x_coord, y_coord, interpolate=None)
    
    response = {
        'edges': f'{nearest_edges}'
    }

    return response, 200

def pathfinder(source, goal, profile, optional_id=None):
    """Main pathfinding function
    Takes 'source', 'goal' and 'profile' as parameters.
    'source' is the source coordinates in [lng,lat]
    'goal' is the destination coordinates in [lng,lat]
    'profile' is the profile of safety factors to take into consideration
    """

    #### SETTINGS ####

    safety_factors = ['not_flood_hazard', 'pwd_friendly',
                      'cctv', 'landmark', 'lighting', 'not_major_road']
    osmnx.settings.useful_tags_way = safety_factors + ['name', 'footway', 'road_closed']

    # comes from application request
    origin = {
        "y": source[0],  # 14.635749969867808,
        "x": source[1]  # 121.09445094913893
    }
    destination = {
        "y": goal[0],  # 14.63056033942939,
        "x": goal[1]  # 121.09807731641334
    }

    params = {
        'lat': source[0],
        'long': source[1],
        'API_key': '998183354bb6d9e4f0bf9a1ce02a8014'
    }

    api_result = requests.get(
        f'https://api.openweathermap.org/data/2.5/weather?lat={params["lat"]}&lon={params["long"]}&appid={params["API_key"]}')

    api_response = api_result.json()

    weather_condition = api_response['weather'][0]['id']

    # retrieve map from database
    # graph = osmnx.graph_from_xml('C:\\Users\\kjqb4\\Documents\\GitHub Projects\\design-project\\Pathfinder_API\\map_complete.osm', simplify=False)
    graph = osmnx.graph_from_xml('map_complete.osm', simplify=False)

    # get all edges for weight adjustment
    nodes, edges = osmnx.graph_to_gdfs(graph)
    edges.sort_index()

    # adjust weights profile depending on user pref and time & weather conditions
    adjusted_profile, conditions = api_profile(weather_condition, profile)

    # adjust safety factors on edges based on reports
    if optional_id:
        edges = report_update_graph(edges, origin, destination, optional_id)
    else:
        edges = report_update_graph(edges, origin, destination)

    # create category "weight" for use in path finding
    edges['weight'] = edges.apply(
        lambda row: adjust_weight(row['length'], row, adjusted_profile), axis=1
    )

    final_graph = osmnx.graph_from_gdfs(
        nodes,
        edges
    )

    origin_node_id = osmnx.nearest_nodes(
        final_graph, origin['x'], origin['y'], return_dist=True)
    destination_node_id = osmnx.nearest_nodes(
        final_graph, destination['x'], destination['y'], return_dist=True)

    # checks if coordinates passed is too far from area covered by map
    if origin_node_id[1] >= 250 or destination_node_id[1] >= 250:
        return {'msg': "Source or destination invalid"}, 400
    else:
        pass

    route = nx.bidirectional_dijkstra(
        final_graph,
        origin_node_id[0],
        destination_node_id[0],
        weight= lambda u, v, attrib: attrib[0]['weight'] if attrib[0]['road_closed'] != '1' else None
    )

    shortest_route = nx.bidirectional_dijkstra(
        final_graph,
        origin_node_id[0],
        destination_node_id[0],
        weight= lambda u, v, attrib: attrib[0]['length'] if attrib[0]['road_closed'] != '1' else None
    )

    route = route[1]
    shortest_route = shortest_route[1]

    route_dir, route_safety_dir = getRouteDirections(
        route, nodes, final_graph, list(adjusted_profile.keys()))

    shortest_route_dir, shortest_route_safety_dir = getRouteDirections(
        shortest_route, nodes, final_graph, list(adjusted_profile.keys()))

    try:
        compare_route = getSafetyFactorCoverage(
            route_safety_dir,
            getRouteLength(route, graph),
            safety_factors,
            adjusted_profile
        )

        compare_shortest_route = getSafetyFactorCoverage(
            shortest_route_safety_dir,
            getRouteLength(shortest_route, graph),
            safety_factors,
            adjusted_profile
        )
    except:
        print(origin, destination)
        return {'msg': "Route is invalid"}, 400

    if compare_route['average'] == compare_shortest_route['average']:
        response = {
            'time': datetime.now(),
            'origin': [origin['y'], origin['x']],
            'destination': [destination['y'], destination['x']],
            'conditions': {'weather': conditions[0], 'lighting': conditions[1]},
            'optimized_route': {
                'coverage': getSafetyFactorCoverage(
                    route_safety_dir,
                    getRouteLength(route, graph),
                    safety_factors,
                    adjusted_profile
                ),
                'length': getRouteLength(route, graph),
                'coordinates': getCoordinates(route, nodes, origin, destination),
                'steps': route_dir
            },
            'shortest_route': {}

        }

        return response, 200
    else:
        response = {
            'time': datetime.now(),
            'origin': [origin['y'], origin['x']],
            'destination': [destination['y'], destination['x']],
            'conditions': {'weather': conditions[0], 'lighting': conditions[1]},
            'optimized_route': {
                'coverage': getSafetyFactorCoverage(
                    route_safety_dir,
                    getRouteLength(route, graph),
                    safety_factors,
                    adjusted_profile
                ),
                'length': getRouteLength(route, graph),
                'coordinates': getCoordinates(route, nodes, origin, destination),
                'steps': route_dir
            },
            'shortest_route': {
                'coverage': getSafetyFactorCoverage(
                    shortest_route_safety_dir,
                    getRouteLength(shortest_route, graph),
                    safety_factors,
                    adjusted_profile
                ),
                'length': getRouteLength(shortest_route, graph),
                'coordinates': getCoordinates(shortest_route, nodes, origin, destination),
                'steps': shortest_route_dir
            }

        }

        return response, 200
