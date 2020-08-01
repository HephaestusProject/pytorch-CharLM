from flask import Flask, request
from flask_restx import Api, Resource
import yaml
import os

environment = os.environ["ENVIRONMENT"]
print(environment)
# with open(f"configs/{environment}.yaml") as y:
#     config = yaml.load(y)

app = Flask(__name__)
api = Api(app)


@api.route("/hello")
class HelloWorld(Resource):
    def get(self):
        return "I am " + greeting("name")


def greeting(key):
    d = {"hello": "hello world", "hi": "hey", "name": config["myname"]}
    return d[key]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
