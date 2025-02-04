import os
import pprint
from email.policy import default

import pandas as pd
import numpy as np
import pathlib as pl
from utils.logger import log

from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix

from utils.project_paths import resources_dir_path

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0.0', title='AI models API', description='AI Enrichment FAIRWork Demonstrator.')

ns = api.namespace('demonstrator', description='Endpoints')

example_order_info_list = []
example_crf_service_input = {}

try:
    import json

    file = resources_dir_path.joinpath("OutputKB_Final.json")
    with open(file) as json_file:
        example_order_info_list = json.load(json_file)["OrderInfoList"]
except Exception as e:
    log.warning(f"could not load example input data: {e}. defaulting to empty list")

try:
    import json

    file = resources_dir_path.joinpath("crf_service_input2.json")
    with open(file) as json_file:
        example_crf_service_input = json.load(json_file)
except Exception as e:
    log.warning(f"could not load example input data: {e}. defaulting to empty list")

worker_preference_list_element = api.model('WorkerPreferenceListElement', {
    'LineId': fields.List(required=True, description='', cls_or_instance=fields.String),
    'Value': fields.List(required=True, description='', cls_or_instance=fields.Float, min=0, max=1),
})

line_info = api.model('LineInfo', {
    'LineId': fields.String(required=True, description=''),
    'Geometry': fields.String(required=True, description=''),
    'ProductionPriority': fields.String(required=True, description='', enum=["True", "False"]),
    'DueDate': fields.Integer(required=True, description='', min=-10, max=10),
    'WorkersRequired': fields.Integer(required=True, description='', min=1, max=8),
})

worker_info_list_element = api.model('WorkerInfoListElement', {
    'Id': fields.String(required=True, description=''),
    'Availability': fields.String(required=True, description='', enum=["True", "False"]),
    'MedicalCondition': fields.String(required=True, description='', enum=["True", "False"]),
    'UTEExperience': fields.String(required=True, description='', enum=["True", "False"]),
    'WorkerResilience': fields.Float(required=True, description='', min=0, max=1),
    'WorkerPreference': fields.List(required=True, description='',
                                    cls_or_instance=fields.Nested(worker_preference_list_element)),
})

order_info = api.model('OrderInfo', {
    'LineInfo': fields.Nested(line_info),
    'WorkerInfoList': fields.List(required=True, description='',
                                  cls_or_instance=fields.Nested(worker_info_list_element)),
})

single_input_service = api.model('InputService', {
    'OrderInfoList': fields.List(
        required=True,
        description='',
        cls_or_instance=fields.Nested(order_info),
        default=example_order_info_list
    ),
})

output_workers_list_element = api.model('OutputWorkersListElement', {
    'Id': fields.String(required=True, description=''),
    'Availability': fields.String(required=True, description='', enum=["True", "False"]),
    'MedicalCondition': fields.String(required=True, description='', enum=["True", "False"]),
    'UTEExperience': fields.String(required=True, description='', enum=["True", "False"]),
    'WorkerResilience': fields.Float(required=True, description='', min=0, max=1),
    'WorkerPreference': fields.Float(required=True, description='', min=0, max=1),
})

output_allocation_list_element = api.model('OutputAllocationListElement', {
    'LineId': fields.String(required=True, description=''),
    'WorkersRequired': fields.Integer(required=True, description='', min=1, max=8),
    'Workers': fields.List(required=True, description='', cls_or_instance=fields.Nested(output_workers_list_element)),
})

single_output_service = api.model('OutputService', {
    'AllocationList': fields.List(required=True, description='',
                                  cls_or_instance=fields.Nested(output_allocation_list_element)),
})

###########################################################################################################################################


"""
{
        "geometry": "533908540",
        "preference": 0.22,
            "resilience": 0.73,
            "medical_condition": "true",
            "experience": 0.33,
            "worker": "15004479"
},
"""

human_factor_model = api.model('HumanFactor', {
    'geometry': fields.String(required=True, example="505597580", description="Geometry name"),
    'preference': fields.Float(required=True, example=0.22, description="Worker preference value", min=0, max=1),
    'resilience': fields.Float(required=True, example=0.73, description="Worker resilience value"),
    'medical_condition': fields.String(required=True, example="true",
                                       description="Indicates if medical conditions exist"),
    'experience': fields.Float(required=True, example=0.9, description="Worker experience value", min=0, max=1),
    'worker': fields.String(required=True, example="15004479", description="Worker identifier")
})

"""
{
            "date": "2023-09-11",
            "from_timestamp": 1694440800,
            "end_timestamp": 1694469600,
            "worker": "15004479"
        },

"""

availabilities_model = api.model('WorkerAvailabilities', {
    'date': fields.String(required=True, example="2023-09-11", description="Availability date (YYYY-MM-DD)"),
    'from_timestamp': fields.Integer(required=True, example=1694440800, description="Start time in Unix timestamp"),
    'end_timestamp': fields.Integer(required=True, example=1694469600, description="End time in Unix timestamp"),
    'worker': fields.String(required=True, example="15004479", description="Worker identifier")
})

"""
{
            "geometry": "534259080",
            "main_line": "line 24",
            "alternative_lines": [
                "line 20"
            ],
            "number_of_workers": 4
        },
"""

geometry_line_mapping_model = api.model('GeometryLineMapping', {
    'geometry': fields.String(required=True, example="534259080", description="Geometry name"),
    'main_line': fields.String(required=True, example="line 24", description="Main production line"),
    'alternative_lines': fields.List(fields.String, required=True, example=["line 17"],
                                     description="Alternative lines"),
    'number_of_workers': fields.Integer(required=True, example=4, description="Number of workers", min=0)
})

"""
{
        "line": "line  20",
        "geometry": "505355480",
        "throughput": 400
},
"""


def validate_throughput(value):
    if not isinstance(value, (str, int)):
        raise ValueError("Throughput must be a string or an integer")


throughput_mapping_model = api.model('ThroughputMapping', {
    'line': fields.String(required=True, example="line 20", description="Production line name"),
    'geometry': fields.String(required=True, example="505355480", description="Geometry name"),
    'throughput': fields.Raw(required=True, example=400, description="Throughput value")
})

"""

{
        "order": "SEV - 38",
        "deadline": 1695362400,
        "priority": "false",
        "geometry": "534259080",
        "amount": 6000,
        "mold": 6
}

"""

order_data_model = api.model('OrderData', {
    'order': fields.String(required=True, example="SEV - 38", description="Order identifier"),
    'deadline': fields.Integer(required=True, example="2021-12-31", description="Order deadline (YYYY-MM-DD)"),
    'priority': fields.String(required=True, example="false", description="Order priority"),
    'geometry': fields.String(required=True, example="534259080", description="Geometry associated with the order"),
    'amount': fields.Integer(required=True, example=6000, description="Order amount"),
    'mold': fields.Integer(required=True, example=4, description="Number of molds")
})

# Combine all models into a single request body model
request_body_model = api.model(
    'WorkerAssignmentRequest', {
        'perform_allocation_for_lines': fields.List(
            fields.String,
            example=["line 17", "line 20", "line 24"],
            default=["line 17", "line 20", "line 24"],
            required=False,
            description='List of lines for which allocation should be performed.'
        ),
        'start_time_timestamp': fields.Integer(
            equired=True,
            example=example_crf_service_input["start_time_timestamp"],
            description="Start time of the planning window in Unix timestamp format."
        ),
        'order_data': fields.List(
            fields.Nested(order_data_model),
            required=True,
            example=example_crf_service_input["order_data"],
            description="List of orders with their details."
        ),
        'geometry_line_mapping': fields.List(
            fields.Nested(geometry_line_mapping_model),
            required=True,
            example=example_crf_service_input["geometry_line_mapping"],
            description="Mapping of geometries to production lines."
        ),
        'throughput_mapping': fields.List(
            fields.Nested(throughput_mapping_model),
            required=True,
            example=[
                {
                    "line": elem["line"],
                    "geometry": elem["geometry"],
                    "throughput": elem["throughput"] if isinstance(elem["throughput"], int) else None
                }
                for elem in example_crf_service_input["throughput_mapping"]
            ],
            description="List of throughput mappings for each line.",
        ),
        'human_factor': fields.List(
            fields.Nested(human_factor_model),
            required=True,
            example=example_crf_service_input["human_factor"],
            description="Human factor details for workers and geometries."
        ),
        'availabilities': fields.List(
            fields.Nested(availabilities_model),
            required=True,
            example=example_crf_service_input["availabilities"],
            description="Availability details for workers."
        ),
    },
)

"""
'start_time_timestamp': fields.Integer(
        equired=True,
        example=1693548000,
        description="Start time of the planning window in Unix timestamp format."
    ),
    'order_data': fields.List(
        fields.Nested(order_data_model),
        required=True,
        description="List of orders with their details."
    ),
    'geometry_line_mapping': fields.List(
        fields.Nested(geometry_line_mapping_model),
        required=True,
        description="Mapping of geometries to production lines."
    ),
    'throughput_mapping': fields.List(
        fields.Nested(throughput_mapping_model),
        required=True,
        description="List of throughput mappings for each line.",
    ),
    'human_factor': fields.List(
        fields.Nested(human_factor_model),
        required=True,
        description="Human factor details for workers and geometries."
    ),
    'availabilities': fields.List(
        fields.Nested(availabilities_model),
        required=True,
        description="Availability details for workers."
    ),"""

"""
        {
            
            "Task": "Order 0",
            "Start": 1699096800000,  // Unix timestamp for "2024-11-04 09:00:00"
            "Finish": 1699104000000, // Unix timestamp for "2024-11-04 13:00:00"
            "Resource": "Line 1",
            "geometry": "geo3",
            "required_workers": 2,
            "workers": [100001, 100002]
        },

"""

output_crf_allocation_list_element = api.model('OutputCrfAllocationListElement', {
    'Task': fields.String(required=True, example="Order `SEV - 35` ; Geometry: `1340746080-8080`",
                          description='A Task is one element, that need to be assigned to resources. It is a unique identifier for the task.'),

    'Start': fields.Integer(required=True, example=1693807200, description='Start time of the Task as Unix timestamp'),
    'Finish': fields.Integer(required=True, example=1693836000, description='End time of the Task as Unix timestamp'),

    'Resource': fields.String(required=True, example="Line 20",
                              description='The Production Line the task is processed on.'),
    'geometry': fields.String(required=True, description='534259080'),

    'required_workers': fields.Integer(required=True, description=''),

    'workers': fields.List(fields.String, required=False, example=["line 17"], description=""),
})

response_crf_body_model = api.model(
    "OutputCrf",
    {
        "experience": fields.Float(required=True, example=0.3,
                                   description="A score that indicates of much the experiences of the workers have for the given task."),
        "preference": fields.Float(required=True, example=0.2,
                                   description="A score that indicates how much the workers prefer the given task."),
        "resilience": fields.Float(required=True, example=0.5,
                                   description="A score that indicates how resilient the workers are for the given task."),
        "transparency": fields.String(required=True, example="medium",
                                      description="A text that indicates how much transparency criteria are met."),

        "allocations": fields.List(required=True, description='',
                                   cls_or_instance=fields.Nested(output_crf_allocation_list_element)),
    },
)

if __name__ == '__main__':
    print('###' * 20)
