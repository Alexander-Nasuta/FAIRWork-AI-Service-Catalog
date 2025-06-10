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
        """
        This service generates a schedule for a given set of orders using a constraint programming (CP) approach.
        It prioritizes high-priority orders, ensuring they are always scheduled before non-priority ones.
        Orders are assigned to production lines in a way that minimizes the makespan—that is, the total time required
        to complete all orders. This service does not consider delivery deadlines.
        Its sole focus is on minimizing makespan, resulting in high machine utilization, even if it means missing some
        deadlines in favor of throughput efficiency.
        """

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
        """
        This service generates a production schedule for a given set of orders using a CP-based approach.
        High-priority orders are always scheduled ahead of others.
        The scheduling goal is to minimize tardiness—defined as the time by which an order misses its deadline.
        Makespan is not taken into account in this optimization. As a result, the system maximizes on-time deliveries,
        even if it reduces machine utilization by prioritizing deadline adherence over throughput.
        """

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
        """
        This service creates a schedule for a given set of orders using a constraint programming strategy that balances
        two objectives: minimizing tardiness and makespan. As with other CRF services, high-priority orders are
        scheduled first.
        The optimization weights tardiness and makespan equally (1:1), meaning that avoiding one hour of tardiness is
        treated as equally valuable as reducing makespan by one hour.
        This balanced approach results in good machine utilization while still avoiding many late deliveries.
        It may occasionally delay early-starting orders in favor of completing others on time to meet due dates more
        effectively.
        """
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

        #print(repr(res))

        return res


@ns.route('/hybrid/cp-balanced/cp-balanced')
class HybridBalancedBalanced(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        '''
        This service implements a two-step constraint programming (CP)-based optimization approach.
        In the first step, the system generates a production schedule by assigning orders to lines while minimizing
        both makespan and tardiness with equal weighting.
        This ensures that deadlines are respected without compromising overall production efficiency, and prioritizes
        urgent orders.

        In the second step, workers are assigned to production lines using a CP-based solver that balances three
        objectives—experience, preference, and resilience—each weighted equally (1:1:1).
        This balanced approach ensures that skilled workers are placed in roles where they are effective, personal
        preferences are respected. The problem is simplified to allow efficient solving while still producing
        high-quality, realistic worker assignments.
        '''
        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-balanced/cp-preference')
class HybridBalancedPreference(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This hybrid service consists of two CP-based optimization stages.
        The first stage uses a CP solver to generate a production schedule that minimizes both makespan and tardiness,
        equally weighted, to balance efficiency with due date adherence. Orders with higher priority are scheduled
        before others.

        The second stage assigns workers based on their preferences using a CP approach.
        The worker assignment process optimizes for maximum satisfaction by taking into account preferred tasks or
        production lines. The problem is simplified to maintain tractable solution times, ensuring both effective
        alignment of personal preferences and feasible execution of generating the production plan.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-balanced/cp-resilience')
class HybridBalancedResilience(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service applies a CP-based two-stage optimization framework.
        The first stage schedules orders to production lines by minimizing makespan and tardiness equally.
        This yields a production plan that is both time-efficient and deadline-sensitive, while also respecting order
        priorities.
        In the second stage, a CP solver allocates workers to lines with a focus worker resilience.
        By simplifying the problem and optimizing for redundancy and skill distribution, the service enhances overall
        operational stability.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-balanced/cp-experience')
class HybridBalancedExperience(Resource):
    @ns.doc('/hybrid/cp-balanced/cp-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step optimization process, both stages using constraint programming (CP).
        In the first step, a CP solver generates a production schedule by assigning orders to lines, optimizing a
        combined objective of makespan and tardiness with equal weighting (1:1). This results in a balanced schedule
        that considers both throughput and delivery punctuality.
        In the second step, workers are assigned to the scheduled lines using a CP-based approach optimized for
        experience.
        The allocation engine prioritizes assigning workers to tasks and lines where their past experience is most
        relevant, while simplifying the underlying optimization problem to ensure reasonable computation times.
        This improves productivity and quality by ensuring skilled personnel are matched with the most suitable tasks.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-balanced/rl-balanced')
class HybridBalancedBalancedRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service optimizes both makespan and tardiness in a two-step process.
        In the first step, a constraint programming (CP) solver generates an optimized production schedule, minimizing
        makespan or tardiness according to the selected target.
        The solver assigns orders to production lines based on the given constraints and priorities.

        In the second step, Reinforcement Learning (RL) is used to allocate workers to the production lines.
        The RL agent has learned an allocation policy based on simulations that consider experience, preference, and
        resilience, each weighted equally (1:1:1).
        The agent applies this learned policy to assign workers in a way that balances all three factors, improving
        the overall productivity and adaptability of the workforce.
        The RL-based allocation approach requires moderate computational resources to generate solution instances,
        but it is still more efficient than some other simulation-based methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='balance')


@ns.route('/hybrid/cp-balanced/rl-preference')
class HybridBalancedPreferenceRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service incorporates a two-step process to manage both tardiness and makespan in production scheduling.
        The first step utilizes a CP solver to generate an optimal schedule, focusing on minimizing tardiness or
        makespanbased on the specified target. This ensures that production is optimized within the constraints of the
        available resources and deadlines.

        In the second phase, Reinforcement Learning (RL) is applied to worker allocation.
        The RL agent has been trained to assign workers to production lines based on worker preferences, ensuring that
        the assignments align with individual worker preferences wherever possible.
        This approach boosts worker satisfaction and morale by considering personal preferences in the scheduling
        process.
        RL-based allocation requires moderate compute resources for training and solution generation, making it a
        balanced approach between complexity and performance.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='preference')


@ns.route('/hybrid/cp-balanced/rl-resilience')
class HybridBalancedResilienceRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step optimization process, starting with a CP solver to generate a schedule that
        minimizes tardiness or makespan, depending on the specific target.
        The CP solver ensures that orders are scheduled efficiently, either minimizing delays or production time while
        maintaining resource availability.

        In the second step, Reinforcement Learning (RL) is used for worker allocation.
        The RL agent is trained to allocate workers based on resilience, ensuring that the allocation helps to maintain
        a stable and adaptable workforce.
        RL-based allocation requires moderate computational resources to process training and generate allocation
        solutions.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='resilience')


@ns.route('/hybrid/cp-balanced/rl-experience')
class HybridBalancedExperienceRL(Resource):
    @ns.doc('/hybrid/cp-balanced/rl-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service leverages a two-step process to efficiently schedule production and allocate workers.
        The first step utilizes a constraint programming (CP) solver to generate an optimized production schedule,
        minimizing either makespan or tardiness, depending on the desired target. The schedule is created with respect
        to resource availability, order priorities, and other constraints.
        In the second phase, Reinforcement Learning (RL) is applied for worker allocation.
        The RL agent learns an allocation policy based on worker experience, which prioritizes assigning more
        experienced workers to tasks where their expertise will add the most value.
        This approach maximizes both productivity and efficiency, ensuring that workers are assigned in a way that best
        utilizes their skillset.
        RL-based allocation requires moderate computational resources for training and generating allocation instances,
        providing a good balance between resource use and performance optimization.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='experience')



@ns.route('/hybrid/cp-balanced/mcts-balanced')
class HybridBalancedBalancedMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step optimization pipeline.
        First, it generates a production schedule using a constraint programming (CP) solver that balances makespan and
        tardiness with equal importance (1:1). This means it considers both timely delivery and efficient machine
        utilization during order-to-line assignment.

        The second step uses Monte Carlo Tree Search (MCTS) to allocate workers to the scheduled lines.
        MCTS explores and simulates many assignment paths, weighing experience, preference, and resilience equally to
        produce a well-rounded and fair allocation. Similar to how a chess player evaluates possible moves ahead of
        time, the MCTS planner anticipates multiple workforce configurations before selecting the most balanced outcome.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-balanced/mcts-preference')
class HybridBalancedPreferenceMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service integrates constraint programming and Monte Carlo Tree Search in a two-step process.
        Initially, it generates a production schedule that optimally balances makespan and tardiness (1:1 ratio),
        prioritizing timely output while maintaining machine efficiency.

        The worker allocation phase uses MCTS to simulate preference-aware assignment paths.
        It actively explores combinations where workers are matched to their preferred roles or stations, and selects
        the most suitable configuration after evaluating multiple outcomes.
        This results in higher satisfaction and engagement while still preserving scheduling feasibility.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-balanced/mcts-resilience')
class HybridBalancedResilienceMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This hybrid service uses constraint programming for production scheduling and Monte Carlo Tree Search for
        workforce assignment.
        The CP-based scheduler aims for a balanced optimization of makespan and tardiness, ensuring that both deadlines
        and operational throughput are respected.
        Worker allocation is then performed using an MCTS strategy that simulates numerous staffing scenarios,
        specifically evaluating their resilience.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-balanced/mcts-experience')
class HybridBalancedExperienceMCTS(Resource):
    @ns.doc('/hybrid/cp-balanced/mcts-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service performs two distinct optimization steps for production planning.
        First, it uses a constraint programming solver to generate a schedule that minimizes makespan, ensuring that
        all orders are completed in the shortest total time, optimizing machine throughput.

        The second phase leverages MCTS for worker allocation, with a focus on maximizing experience-based fit.
        The MCTS engine simulates various assignment options and prioritizes solutions where workers with the most
        relevant experience are matched to appropriate lines. This promotes operational quality and leverages the
        workforce’s skill base effectively.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
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
        """
        This service uses a two-step optimization process driven by constraint programming (CP).
        In the first step, the service creates a schedule that minimizes makespan—the total production time needed to
        complete all orders. The scheduling logic ensures high machine utilization, with priority orders handled before
        others, but it does not consider tardiness or deadlines.

        In the second step, workers are assigned using a CP solver that balances three human-centric objectives:
        experience, preference, and resilience—each with equal weighting. The allocation problem is simplified to reduce
        computation time while ensuring the final solution is practical and fair, placing capable, satisfied, and
        reliable workers across the production lines.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-makespan/cp-preference')
class HybridMakespanPreference(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service uses a constraint programming-based two-step approach.
        In the first step, orders are scheduled to minimize makespan, maximizing production throughput and prioritizing
        urgent orders, without considering order deadlines.

        The second step uses CP to allocate workers with an emphasis on individual preferences—placing workers where
        they prefer to be assigned whenever feasible. The model is simplified to remain computationally efficient while
        delivering a solution that supports satisfaction and motivation in the workforce.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-makespan/cp-resilience')
class HybridMakespanResilience(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service leverages a two-step constraint programming methodology.
        First, it generates a schedule minimizing makespan, enabling high line efficiency and fast order completion,
        with a strict priority on urgent orders and no consideration of deadlines.

        The second step allocates workers via a CP solver focused on resilience. The solver reduces complexity to
        ensure timely results, delivering a robust and practical workforce layout.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-makespan/cp-experience')
class HybridMakespanExperience(Resource):
    @ns.doc('/hybrid/cp-makespan/cp-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step CP-based optimization process.
        In the first step, it generates a schedule that strictly minimizes the makespan, focusing on reducing total
        production duration while always prioritizing urgent orders. Deadlines are not considered.

        The second step involves CP-based worker assignment optimized for experience—ensuring that workers are placed
        where their past performance and skills best match the task.
        The solver simplifies the assignment space to produce results quickly while still making effective use of
        workforce expertise.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-makespan/rl-balanced')
class HybridMakespanBalancedRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step approach to optimize production scheduling and worker allocation.
        In the first step, an exact constraint programming (CP) solver is used to generate an optimized schedule.
        The solver minimizes the makespan or tardiness based on the target set for the production process,
        considering priority orders and available resources.

        In the second step, Reinforcement Learning (RL) is applied to allocate workers to production lines.
        The RL model learns a policy based on random preference and resilience scores, with a balanced weighting of
        1:1:1.
        This means the model takes into account experience, preference, and resilience to assign workers in a manner
        that ensures both optimal machine utilization and workforce satisfaction.
        RL-based allocation requires moderate compute resources to generate solution instances, balancing the complexity
        of learning with practical performance.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='balance')


@ns.route('/hybrid/cp-makespan/rl-preference')
class HybridMakespanPreferenceRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service implements a two-step process to optimize both makespan and tardiness while considering worker
        preferences.
        In the first step, an exact CP solver generates a production schedule that minimizes the makespan or tardiness,
        depending on the chosen target.
        The solver takes into account various constraints, such as order priorities and available production lines,
        to create an optimized schedule.

        The second step uses Reinforcement Learning (RL) to allocate workers to production lines.
        The RL agent learns an allocation policy that prioritizes worker preferences, ensuring that employees are
        assigned to tasks they prefer.
        This approach improves worker satisfaction and morale while still striving to meet the operational objectives of
        the production process.
        RL-based allocation requires moderate computational power, offering a practical solution for balancing worker
        satisfaction with production efficiency.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='preference')


@ns.route('/hybrid/cp-makespan/rl-resilience')
class HybridMakespanResilienceRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service utilizes a two-step process for optimizing both makespan and tardiness in production scheduling.
        The first step involves a CP solver that generates an optimal production schedule, minimizing either tardiness
        or makespan, based on the selected target.
        This ensures that all orders are completed within the desired timeframe and resources are utilized efficiently.

        The second step applies Reinforcement Learning (RL) for worker allocation.
        The RL agent is trained to consider worker resilience when assigning workers to production lines.
        RL-based allocation requires moderate computational resources, balancing the need for efficient learning with
        the complexity of resilience-based worker allocation.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='resilience')


@ns.route('/hybrid/cp-makespan/rl-experience')
class HybridMakespanExperienceRL(Resource):
    @ns.doc('/hybrid/cp-makespan/rl-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service leverages a two-step process to efficiently schedule production and allocate workers.
        The first step utilizes a constraint programming (CP) solver to generate an optimized production schedule,
        minimizing either makespan or tardiness, depending on the desired target.
        The schedule is created with respect to resource availability, order priorities, and other constraints.

        In the second phase, Reinforcement Learning (RL) is applied for worker allocation.
        The RL agent learns an allocation policy based on worker experience, which prioritizes assigning more
        experienced workers to tasks where their expertise will add the most value.
        This approach maximizes both productivity and efficiency, ensuring that workers are assigned in a way that best
        utilizes their skillset.
        RL-based allocation requires moderate computational resources for training and generating allocation instances,
        providing a good balance between resource use and performance optimization.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='experience')



@ns.route('/hybrid/cp-makespan/mcts-balanced')
class HybridMakespanBalancedMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service applies a two-step optimization pipeline that begins with a CP-based scheduling of orders to
        production lines.
        The scheduling phase focuses exclusively on minimizing makespan, ensuring that the total production time across
        all lines is as short as possible, which leads to high machine utilization but may disregard individual order
        deadlines.

        The second step uses Monte Carlo Tree Search (MCTS) to assign workers to the scheduled production lines.
        MCTS explores many possible allocation paths and simulates outcomes by evaluating a balanced objective, where
        experience, preference, and resilience are weighted equally (1:1:1).
        This leads to fair, robust, and satisfaction-aware assignments, mimicking strategic thinking similar to a chess
        player’s planning of moves.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-makespan/mcts-preference')
class HybridMakespanPreferenceMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service begins by generating a schedule using a CP solver that is dedicated to minimizing makespan,
        reducing total processing time and maximizing equipment usage.
        This exact solver produces a highly compact schedule based solely on production efficiency.

        Worker assignment is then carried out using MCTS, which simulates different allocation outcomes and prioritizes
        configurations that respect individual worker preferences.
        This leads to improved satisfaction and team morale while still ensuring assignments remain feasible and
        consistent with the production schedule.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-makespan/mcts-resilience')
class HybridMakespanResilienceMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This two-phase service starts by generating a schedule via a constraint programming solver that focuses purely
        on makespan minimization.
        This ensures that all orders are processed as quickly as possible, with priority given to machine utilization
        rather than delivery deadlines.
        In the second step, MCTS is used to assign workers to production lines with a strong emphasis on resilience.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-makespan/mcts-experience')
class HybridMakespanExperienceMCTS(Resource):
    @ns.doc('/hybrid/cp-makespan/mcts-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service performs two distinct optimization steps for production planning.
        First, it uses a constraint programming solver to generate a schedule that minimizes makespan, ensuring that
        all orders are completed in the shortest total time, optimizing machine throughput.

        The second phase leverages MCTS for worker allocation, with a focus on maximizing experience-based fit.
        The MCTS engine simulates various assignment options and prioritizes solutions where workers with the most
        relevant experience are matched to appropriate lines. This promotes operational quality and leverages the
        workforce’s skill base effectively.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
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
        """
        This service performs a two-step optimization using constraint programming.
        In the first step, it creates a production schedule that minimizes total tardiness, aiming to complete as many
        orders as possible before their deadlines, while still prioritizing urgent orders.

        The second step assigns workers using a CP-based solver that balances experience, preference, and
        resilience—each given equal weight.
        The assignment model is simplified to remain computationally tractable while producing allocations that ensure
        a reliable, skilled, and satisfied workforce.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-tardiness/cp-preference')
class HybridMakespanPreference(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service uses constraint programming in both scheduling and worker allocation.
        In the first step, it builds a schedule that minimizes tardiness, helping to maximize on-time order completion
        while maintaining priority handling.
        The second step leverages a CP solver that respects worker preferences—aiming to match individuals to their
        favored roles or production lines.
        The model simplifies constraints for faster computation while ensuring a level of assignment personalization
        that can boost morale and engagement.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-tardiness/cp-resilience')
class HybridMakespanResilience(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This endpoint uses a two-step constraint programming approach.
        In the first stage, it creates a schedule that aims to minimize order tardiness, ensuring deadlines are met as
        effectively as possible and prioritizing urgent orders.

        The second stage focuses on resilience in worker allocation.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-tardiness/cp-experience')
class HybridMakespanExperience(Resource):
    @ns.doc('/hybrid/cp-tardiness/cp-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service applies a two-step CP-driven optimization.
        The first step generates a schedule focused exclusively on minimizing tardiness, ensuring that as few orders
        as possible are completed late and prioritizing urgent ones when applicable.

        In the second step, workers are allocated through a CP model that emphasizes experience, matching staff to
        production lines based on their past performance and skill history.
        To keep computation time low, the solver uses a simplified formulation while still leveraging expertise
        effectively across the system.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_cp(env, api_payload=api.payload)


@ns.route('/hybrid/cp-tardiness/rl-balanced')
class HybridMakespanBalancedRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step process to optimize production scheduling and worker allocation.
        In the first step, an exact constraint programming (CP) solver is used to generate a production schedule that
        minimizes tardiness.
        The solver considers the available resources, the priority of orders, and other constraints to ensure the most
        efficient use of time while reducing delays.

        In the second step, Reinforcement Learning (RL) is used to allocate workers to the production lines.
        The RL model is trained to learn a policy based on random preference and resilience scores.
        The model assigns workers to lines in a manner that considers their experience, preferences, and adaptability
        to changes, with a balanced weighting of 1:1:1 for all three factors.
        This ensures that worker assignments are both effective and fair, helping to achieve high productivity while
        maintaining worker satisfaction and resilience.
        RL-based allocation requires moderate computational power to generate solution instances, offering a balanced
        approach to worker assignment and performance optimization.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='balance')


@ns.route('/hybrid/cp-tardiness/rl-preference')
class HybridMakespanPreferenceRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service utilizes a two-step process to optimize tardiness while considering worker preferences.
        The first step involves an exact CP solver that generates a production schedule based on minimizing tardiness.
        The solver ensures that orders are completed within their required timeframes while taking into account
        available resources and order priorities.

        In the second step, Reinforcement Learning (RL) is employed for worker allocation.
        The RL agent learns a policy that optimally assigns workers based on their preferences for specific tasks or
        shifts.
        By considering worker preferences, the system enhances job satisfaction, which can lead to improved morale and
        productivity.
        RL-based allocation requires moderate computational power to generate solution instances, but it strikes a good
        balance between personalization and production efficiency.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='preference')


@ns.route('/hybrid/cp-tardiness/rl-resilience')
class HybridMakespanResilienceRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step process for optimizing tardiness while ensuring resilient worker allocation.
        The first step uses an exact CP solver to generate an optimized production schedule, with a focus on minimizing
        tardiness.
        This ensures that orders are delivered on time, with the solver taking into account all necessary constraints,
        such as resource availability and order priorities.

        In the second step, Reinforcement Learning (RL) is applied to allocate workers to the production lines.
        The RL model learns a policy based on resilience factors.
        RL-based allocation requires moderate computational resources, providing a balanced solution that enhances both
        worker resilience and operational performance.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='resilience')


@ns.route('/hybrid/cp-tardiness/rl-experience')
class HybridMakespanExperienceRL(Resource):
    @ns.doc('/hybrid/cp-tardiness/rl-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service optimizes the production process with a two-step approach, ensuring minimal tardiness and effective
        worker allocation.
        In the first step, an exact CP solver generates a schedule that minimizes tardiness by considering various
        constraints, such as production priorities, worker availability, and the resources at hand.

        In the second step, Reinforcement Learning (RL) is applied to allocate workers to production lines.
        The RL agent learns a policy based on worker experience, prioritizing the assignment of tasks to workers who
        have the necessary skills and expertise to handle them efficiently.
        RL-based allocation requires moderate computational resources for training and generating allocation instances,
        balancing learning time with optimization benefits.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_rl(env, focus='experience')



@ns.route('/hybrid/cp-tardiness/mcts-balanced')
class HybridMakespanBalancedMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-balanced')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service applies a two-step optimization process for scheduling and worker allocation.
        In the first step, it uses a constraint programming (CP) solver to generate an optimal production schedule,
        focusing on minimizing tardiness—the delay in completing orders beyond their deadlines.
        The CP-based scheduling ensures that all orders are completed as soon as possible, improving overall delivery
        performance.

        The second phase uses Monte Carlo Tree Search (MCTS) for worker assignment, simulating multiple allocation
        possibilities to identify the best fit. The MCTS approach evaluates balanced objectives by considering
        experience, preference, and resilience with equal weighting (1:1:1), ensuring fairness and adaptability across
        the workforce. The solution takes a strategic approach, similar to a chess player considering future moves
        before finalizing decisions.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-tardiness/mcts-preference')
class HybridMakespanPreferenceMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-preference')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service first utilizes a constraint programming (CP) approach to generate a schedule that minimizes
        tardiness, focusing on completing orders as close to their deadlines as possible while reducing delays.
        The solver produces an efficient schedule that optimizes production time in line with the available resources.

        In the second step, Monte Carlo Tree Search (MCTS) is used to allocate workers to the production lines.
        The MCTS engine simulates various worker assignment scenarios, with an emphasis on meeting worker preferences.
        This ensures that workers are matched with lines that align with their individual preferences, contributing to
        higher job satisfaction and more stable production performance.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-tardiness/mcts-resilience')
class HybridMakespanResilienceMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-resilience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service follows a two-step process, starting with constraint programming (CP) to generate a production
        schedule that minimizes tardiness by prioritizing the completion of orders within their deadlines.
        The CP-based approach ensures an optimal use of available resources while minimizing delays and managing
        workload effectively.

        The second phase uses Monte Carlo Tree Search (MCTS) to allocate workers to the production lines.
        The MCTS approach evaluates different worker assignments, prioritizing resilience in the allocation process.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
        )

        return solve_with_mcts(env, n_sim=20)


@ns.route('/hybrid/cp-tardiness/mcts-experience')
class HybridMakespanExperienceMCTS(Resource):
    @ns.doc('/hybrid/cp-tardiness/mcts-experience')
    @ns.expect(request_body_model)
    @ns.marshal_with(response_crf_body_model)
    def post(self):
        """
        This service integrates constraint programming (CP) in the first step to create a schedule that minimizes
        tardiness, ensuring that orders are completed as quickly as possible, with particular focus on avoiding overdue
        times. This CP solver considers the constraints of each order’s deadline and the available resources to generate
        an optimized solution for the production line.

        In the second phase, the service applies Monte Carlo Tree Search (MCTS) for worker allocation.
        The MCTS algorithm simulates different assignment options, with a focus on matching workers based on experience.
        This approach ensures that workers are assigned to production lines in a way that leverages their skills and
        expertise, leading to improved efficiency and smoother production processes.
        MCTS approaches are computationally heavy and result in the longest solution times among all allocation methods.
        """

        api_data = api.payload

        start_timestamp = api_data["start_time_timestamp"]

        worker_availabilities = api_data["availabilities"]
        geometry_line_mapping = api_data["geometry_line_mapping"]
        human_factor_data = api_data["human_factor"]
        order_data = api_data["order_data"]
        throughput_mapping = api_data["throughput_mapping"]

        # change start timestamp to the earliest worker availability if there is a mismatch
        start_timestamp = max(
            start_timestamp,
            min([elem['from_timestamp'] for elem in worker_availabilities])
        ) if len(worker_availabilities) else start_timestamp

        # return empy list if there are no orders
        if not len(order_data):
            return {
                "experience": 0,
                "preference": 0,
                "resilience": 0,
                "transparency": "high",
                "allocations": [],
            }

        allocations_dict = _perform_order_to_line_mapping(
            api_payload=api_data,
            start_time_timestamp=start_timestamp,
            makespan_weight=1,
            tardiness_weight=1
        )

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

            order_data=order_data,
            throughput_mapping=throughput_mapping,
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
