import os

from flask_restx import Resource, abort

from demonstrator.linear_assignment_solver import allocate_using_linear_assignment_solver
from demonstrator.neural_network import get_solution
from rest.flaskx_api_namespace import ns, api, app, output_service, input_service
from utils.logger import log
from validation.input_validation import validate_instance
from validation.output_validation import validate_output_dict


@ns.route('/nn/')
class NeuralNetwork(Resource):

    @ns.doc('allocate-orders')
    @ns.expect(input_service)
    @ns.marshal_list_with(output_service)
    def post(self):
        """ Endpoint for the neural network model."""
        instance = api.payload
        log.info(f"received instance", extra=instance)

        try:
            validate_instance(instance)
        except (TypeError, ValueError, KeyError) as e:
            log.error(f"instance validation failed: {e}", extra=instance)
            abort(400, f'Invalid payload: {e}')

        service_output = get_solution(instance=instance)

        validate_output_dict(service_output)
        service_output = allocate_using_linear_assignment_solver(instance)
        return service_output


@ns.route('/linear-assignment-optimizer/')
class LinearAssignmentOptimizer(Resource):

    @ns.doc('allocate-orders')
    @ns.expect(input_service)
    @ns.marshal_list_with(output_service)
    def post(self):
        """ Endpoint for the linear assignment optimizer."""
        instance = api.payload
        log.info(f"received instance", extra=instance)
        try:
            validate_instance(instance)
        except (TypeError, ValueError, KeyError) as e:
            log.error(f"instance validation failed: {e}", extra=instance)
            abort(400, f'Invalid payload: {e}')
        service_output = allocate_using_linear_assignment_solver(instance)
        log.info(f"sending response.", extra=service_output)
        return service_output


def import_endpoints():
    return app


def main() -> None:
    log.info("starting flask app...")
    app.run(port=8080)


if __name__ == '__main__':
    main()
