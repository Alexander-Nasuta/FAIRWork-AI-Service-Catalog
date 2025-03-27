import pprint
import itertools
import numpy as np
from typing import List, Dict, Any

import pandas as pd

from flask_restx import Resource, abort
from jsp_vis.console import gantt_chart_console

from demonstrator.algo_collection import _perform_order_to_line_mapping, solve_with_cp, solve_with_rl, solve_with_mcts
from demonstrator.crf_worker_allocation_gym_env import CrfWorkerAllocationEnv
from demonstrator.linear_assignment_solver import allocate_using_linear_assignment_solver
from demonstrator.neural_network import get_solution
from rest.flaskx_api_namespace_crf import ns, api, app, single_output_service, single_input_service, human_factor_model, \
    availabilities_model, geometry_line_mapping_model, throughput_mapping_model, order_data_model
from rest.flaskx_api_namespace_crf import request_body_model, response_crf_body_model
from utils.crf_timestamp_solver_time_conversion import log_time, timestamp_to_solver_time, solver_time_to_timestamp

from utils.logger import log
from validation.input_validation import validate_instance
from validation.output_validation import validate_output_dict

from demonstrator.crf_order_to_line_cp_solver import main as crf_solve_order_to_line_instance


@ns.route('/nn/')
class NeuralNetwork(Resource):

    @ns.doc('/nn/')
    @ns.expect(single_input_service)
    @ns.marshal_list_with(single_output_service)
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

    @ns.doc('/linear-assignment-optimizer/')
    @ns.expect(single_input_service)
    @ns.marshal_list_with(single_output_service)
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


@ns.route('/crf-order-to-line/optimize-makespan')
class CrfCpOptimizerMakespanEndpoint(Resource):
    @ns.doc('/crf-order-to-line/optimize-makespan')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint for the linear assignment optimizer."""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        return {
            "experience": None,
            "preference": None,
            "resilience": None,
            "transparency": "high",
            "allocations": allocations_dict,
        }


@ns.route('/crf-order-to-line/optimize-tardiness')
class CrfCpOptimizerTardinessEndpoint(Resource):
    @ns.doc('/crf-order-to-line/optimize-tardiness')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint """

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        return {
            "experience": None,
            "preference": None,
            "resilience": None,
            "transparency": "high",
            "allocations": allocations_dict,
        }


@ns.route('/crf-order-to-line/optimize-tardiness-and-makespan')
class CrfCpOptimizerTardinessAndMakespanEndpoint(Resource):
    @ns.doc('/crf-order-to-line/optimize-tardiness-and-makespan')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""
        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        res = {
            "experience": None,
            "preference": None,
            "resilience": None,
            "transparency": "high",
            "allocations": allocations_dict,
        }

        print(repr(res))

        return res


@ns.route('/hybrid/cp-balanced/cp-balanced')
class HybridBalancedBalanced(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-balanced/cp-preference')
class HybridBalancedPreference(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint for the linear assignment optimizer."""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=0,
            experience_weight=0,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-balanced/cp-resilience')
class HybridBalancedResilience(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=1,
            experience_weight=0,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-balanced/cp-experience')
class HybridBalancedExperience(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=0,
            experience_weight=1,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-balanced/rl-balanced')
class HybridBalancedBalancedRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=2,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='balance')


@ns.route('/hybrid/cp-balanced/rl-preference')
class HybridBalancedPreferenceRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=10,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='preference')


@ns.route('/hybrid/cp-balanced/rl-resilience')
class HybridBalancedResilienceRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=10,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='resilience')


@ns.route('/hybrid/cp-balanced/rl-experience')
class HybridBalancedExperienceRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=10,
        )

        return solve_with_rl(env, focus='experience')



@ns.route('/hybrid/cp-balanced/mcts-balanced')
class HybridBalancedBalancedMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-balanced/mcts-preference')
class HybridBalancedPreferenceMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=0,
            experience_weight=0,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-balanced/mcts-resilience')
class HybridBalancedResilienceMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=1,
            experience_weight=0,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-balanced/mcts-experience')
class HybridBalancedExperienceMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=0,
            experience_weight=1,
        )

        return solve_with_mcts(env, n_sim=20)


########################################################################################################################
########################################################################################################################
########################################################################################################################

@ns.route('/hybrid/cp-makespan/cp-balanced')
class HybridMakespanBalanced(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-makespan/cp-preference')
class HybridMakespanPreference(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint for the linear assignment optimizer."""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=0,
            experience_weight=0,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-makespan/cp-resilience')
class HybridMakespanResilience(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=1,
            experience_weight=0,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-makespan/cp-experience')
class HybridMakespanExperience(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=0,
            experience_weight=1,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-makespan/rl-balanced')
class HybridMakespanBalancedRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=2,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='balance')


@ns.route('/hybrid/cp-makespan/rl-preference')
class HybridMakespanPreferenceRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=10,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='preference')


@ns.route('/hybrid/cp-makespan/rl-resilience')
class HybridMakespanResilienceRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=10,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='resilience')


@ns.route('/hybrid/cp-makespan/rl-experience')
class HybridMakespanExperienceRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=10,
        )

        return solve_with_rl(env, focus='experience')



@ns.route('/hybrid/cp-makespan/mcts-balanced')
class HybridMakespanBalancedMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-makespan/mcts-preference')
class HybridMakespanPreferenceMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=0,
            experience_weight=0,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-makespan/mcts-resilience')
class HybridMakespanResilienceMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=1,
            experience_weight=0,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-makespan/mcts-experience')
class HybridMakespanExperienceMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=1,
            tardiness_weight=0
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=0,
            experience_weight=1,
        )

        return solve_with_mcts(env, n_sim=20)

########################################################################################################################
########################################################################################################################
########################################################################################################################


@ns.route('/hybrid/cp-tardiness/cp-balanced')
class HybridMakespanBalanced(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-tardiness/cp-preference')
class HybridMakespanPreference(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint for the linear assignment optimizer."""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=0,
            experience_weight=0,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-tardiness/cp-resilience')
class HybridMakespanResilience(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=1,
            experience_weight=0,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-tardiness/cp-experience')
class HybridMakespanExperience(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=0,
            experience_weight=1,
        )

        return solve_with_cp(env)


@ns.route('/hybrid/cp-tardiness/rl-balanced')
class HybridMakespanBalancedRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=2,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='balance')


@ns.route('/hybrid/cp-tardiness/rl-preference')
class HybridMakespanPreferenceRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=10,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='preference')


@ns.route('/hybrid/cp-tardiness/rl-resilience')
class HybridMakespanResilienceRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=10,
            experience_weight=1,
        )

        return solve_with_rl(env, focus='resilience')


@ns.route('/hybrid/cp-tardiness/rl-experience')
class HybridMakespanExperienceRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=10,
        )

        return solve_with_rl(env, focus='experience')



@ns.route('/hybrid/cp-tardiness/mcts-balanced')
class HybridMakespanBalancedMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=1,
            experience_weight=1,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-tardiness/mcts-preference')
class HybridMakespanPreferenceMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=1,
            resilience_weight=0,
            experience_weight=0,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-tardiness/mcts-resilience')
class HybridMakespanResilienceMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=1,
            experience_weight=0,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-tardiness/mcts-experience')
class HybridMakespanExperienceMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """ Endpoint"""

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api.payload,
            makespan_weight=0,
            tardiness_weight=1
        )

        start_timestamp = api.payload["start_time_timestamp"]

        worker_availabilities = api.payload["availabilities"]
        geometry_line_mapping = api.payload["geometry_line_mapping"]
        human_factor_data = api.payload["human_factor"]

        env = CrfWorkerAllocationEnv(
            previous_step_output=allocations_dict,
            worker_availabilities=worker_availabilities,
            geometry_line_mapping=geometry_line_mapping,
            human_factor_data=human_factor_data,
            start_timestamp=start_timestamp,
            allocate_workers_on_the_same_line_if_possible=False,
            preference_weight=0,
            resilience_weight=0,
            experience_weight=1,
        )

        return solve_with_mcts(env, n_sim=20)

########################################################################################################################




def import_endpoints():
    return app


def main() -> None:
    log.info("starting flask app...")
    app.run(
        port=8080,
        threaded=True
    )


if __name__ == '__main__':
    main()
