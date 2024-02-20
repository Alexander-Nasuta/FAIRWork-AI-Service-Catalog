import os

import pandas as pd
import numpy as np
import pathlib as pl
from utils.logger import log

from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0.0', title='AI models API', description='AI Enrichment FAIRWork Demonstrator.')

ns = api.namespace('demonstrator', description='Endpoint calls')

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
    'WorkerInfoList': fields.List(required=True, description='', cls_or_instance=fields.Nested(worker_info_list_element)),
})

input_service = api.model('InputService', {
    'OrderInfoList': fields.List(required=True, description='', cls_or_instance=fields.Nested(order_info)),
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

output_service = api.model('OutputService', {
    'AllocationList': fields.List(required=True, description='', cls_or_instance=fields.Nested(output_allocation_list_element)),
})

