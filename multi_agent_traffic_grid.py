"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.core import rewards as z
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
from flow.envs.multiagent import MultiEnv
from copy import deepcopy

ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiTrafficLightGridPOEnv(TrafficLightGridPOEnv, MultiEnv):
    """Multiagent shared model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)
        #self.min_switch_time= 50

        self.at_light = {rl_id_num: [{}, {}, {}, {}] for 
            rl_id_num, _ in self.network.node_mapping} # IDs of cars currently on each edge and their wait times
        self.on_edge = {rl_id_num: [[], [], [], []] for 
            rl_id_num, _ in self.network.node_mapping}


    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, local edge information, and traffic
        light state.
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(3 * 4 * self.num_observed +
                   2 * self.num_local_edges +
                   2 * (1 + self.num_local_lights),
                   ),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection, edge number.
        - For edges in the network, gives the length of the queue, # of vehicles,
          the average waiting time, the delay of each vehicle
        """
        # Normalization factors
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # TODO(cathywu) refactor TrafficLightGridPOEnv with convenience
        # methods for observations, but remember to flatten for single-agent

        # Observed vehicle information
        obs = {}
        for rl_id_num, edges in self.network.node_mapping:
            local_queue = []
            local_num_vehicles = []
            local_delays = []
            local_wait_time = []
            local_direction = self.direction[int(rl_id_num[-1])]
            for i, edge in enumerate(edges):
                observed_ids = \
                    self.get_closest_to_intersection(edge, 100)

                # Get Queue length (Speed of less than 0.1 m/s )
                queue_length = sum([int(-1 <= self.k.vehicle.get_speed(veh_id) <= 1) for
                    veh_id in observed_ids])

                # Get number of vehicles in each lane
                num_vehicle = sum([int(veh_id != "") for veh_id in observed_ids])

                # Delay of each vehicle
                avg_speed = sum([self.k.vehicle.get_speed(veh_id) for veh_id in observed_ids
                            if veh_id != ""]) / (num_vehicle + 1e-5)
                delay = 1 - avg_speed / max_speed

                # Calculate average wait time of each edge
                if queue_length == 0:
                    wait_time = 0
                    self.at_light[rl_id_num][i] = {}
                else:
                    old_vehicles = self.at_light[rl_id_num][i]
                    observed_stopped = [veh_id for veh_id in observed_ids
                               if -0.1 <= self.k.vehicle.get_speed(veh_id) <= 0.1]

                    intersected_vehicles = list(set(observed_stopped) & set(old_vehicles.keys()))

                    self.at_light[rl_id_num][i] = {
                        veh_id: self.at_light[rl_id_num][i][veh_id] + 1 * self.sim_step
                        for veh_id in intersected_vehicles}

                    for veh_id in observed_stopped:
                        if veh_id not in intersected_vehicles:
                            self.at_light[rl_id_num][i][veh_id] = 1 * self.sim_step

                    wait_time = sum(self.at_light[rl_id_num][i].values()) / len([i for i in observed_ids if i != ""])


                local_queue.append(queue_length)
                local_num_vehicles.append(num_vehicle)
                local_delays.append(delay)
                local_wait_time.append(wait_time)

            observation = np.array(np.concatenate(
                [local_queue, local_wait_time, local_delays, local_num_vehicles, 
                local_direction]))
            

            obs.update({rl_id_num: observation})

        return obs


    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues action for each traffic light agent.
        """
        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
            if self.discrete:
                raise NotImplementedError
            else:
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action[0] > 0.0

            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= 3:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='rGrG')
                    self.currently_yellow[i] = 0
                    self.last_change[i] = 0
            else:
                if action and self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1
                else:
                    '''
                    My code to make sure that the traffic light stage does not 
                    change unless instructed. have to make sure that last_change
                    is not affected
                    '''
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='GrGr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='rGrG')
                    self.last_change[i] += self.sim_step

    def compute_individual_reward   (self, rl_actions, state_dict, last_change):
        '''
        Function to compute the reward of an individual traffic light
        as in https://arxiv.org/pdf/1909.10651.pdf

        Attributes
        ----------
        - rl_actions : TrafficAction object from Env.py
        - state_dict : What is returned from get_state
        - last_change : last change of each light

        Returns
        -------
        - rewards : Dictionary of individual rewards [num_lights]
        '''
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        action_sum = sum([int(rl_actions["center{}".format(k)]) for k in range(len(last_change))])
        # First we need to calculate the number of vehicles that pass each light
        # and the number of emergency stops that have occured
        rewards = {}
        j = 0
        for rl_id_num, edges in self.network.node_mapping:
            total_num_passed = 0
            total_em_stops = 0
            total_wait = 0
            total_queue = 0
            switched = int(self.last_change[int(rl_id_num[-1])][0] == 0 and self.currently_yellow[int(rl_id_num[-1])][0] != 0)
            for i, edge in enumerate(edges):
                observed_ids = \
                    self.get_closest_to_intersection(edge, 100)
                queue_length = sum([int(-1 <= self.k.vehicle.get_speed(veh_id) <= 1) for
                    veh_id in observed_ids])

                observed_ids = [veh_id for veh_id in observed_ids if veh_id != ""]
                # Find cars who stayed on edge for two turns
                intersected_cars = list(set(self.on_edge[rl_id_num][i]) & set(observed_ids))
                # Calculate the number of cars that passed through
                num_passed = len(self.on_edge[rl_id_num][i]) - len(intersected_cars)
                # Update the cars currently on the edge
                self.on_edge[rl_id_num][i] = observed_ids
                total_num_passed += num_passed

                # Calculate the number of emergency stops/brakes
                accels = [
                        (self.k.vehicle.get_speed(veh_id) - self.k.vehicle.get_previous_speed(veh_id)) / self.k.vehicle.get_timedelta(veh_id)
                        for veh_id in observed_ids]
                emergencies = [a for a in accels if a <= -4.5]
                total_em_stops += len(emergencies)
                total_queue += queue_length


            # Next we calculate the sum of vehicle wait times
            wait_times = state_dict[rl_id_num][4:8]
            total_wait = sum(wait_times) 

            # # Next we calculate the total delay
            delays = state_dict[rl_id_num][8:12]
            total_delay = sum(delays)

            # Now we compute the weighted reward
            reward = -0.5 * total_queue + -0.5 * total_wait - 0.5 * total_delay + total_num_passed - 0.25 * total_em_stops - switched
            # reward = -0.5 * total_queue

            rewards[rl_id_num] = reward

            j+=1

        return rewards

    def compute_reward_stats(self, rl_actions, state_dict, last_change):
        '''
        Function to compute the reward of an individual traffic light
        as in https://arxiv.org/pdf/1909.10651.pdf

        Attributes
        ----------
        - rl_actions : TrafficAction object from Env.py
        - state_dict : What is returned from get_state
        - last_change : last change of each light

        Returns
        -------
        - rewards : Dictionary of individual rewards [num_lights]
        '''
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # First we need to calculate the number of vehicles that pass each light
        # and the number of emergency stops that have occured
        rewards = {}
        j = 0
        global_queue = 0
        global_wait = 0
        global_delay = 0
        global_switched = 0
        global_passed = 0
        global_em = 0
        for rl_id_num, edges in self.network.node_mapping:
            total_num_passed = 0
            total_em_stops = 0
            total_wait = 0
            total_queue = 0
            switched = int(self.last_change[int(rl_id_num[-1])][0] == 0 and self.currently_yellow[int(rl_id_num[-1])][0] != 0)
            for i, edge in enumerate(edges):
                observed_ids = \
                    self.get_closest_to_intersection(edge, 100)
                queue_length = sum([int(-1 <= self.k.vehicle.get_speed(veh_id) <= 1) for
                    veh_id in observed_ids])

                observed_ids = [veh_id for veh_id in observed_ids if veh_id != ""]
                # Find cars who stayed on edge for two turns
                intersected_cars = list(set(self.on_edge[rl_id_num][i]) & set(observed_ids))
                # Calculate the number of cars that passed through
                num_passed = len(self.on_edge[rl_id_num][i]) - len(intersected_cars)
                # Update the cars currently on the edge
                self.on_edge[rl_id_num][i] = observed_ids
                total_num_passed += num_passed

                # Calculate the number of emergency stops/brakes
                accels = [
                        (self.k.vehicle.get_speed(veh_id) - self.k.vehicle.get_previous_speed(veh_id)) / self.k.vehicle.get_timedelta(veh_id)
                        for veh_id in observed_ids]
                emergencies = [a for a in accels if a <= -4.5]
                total_em_stops += len(emergencies)
                total_queue += queue_length


            # Next we calculate the sum of vehicle wait times
            wait_times = state_dict[rl_id_num][4:8]
            total_wait = sum(wait_times) 

            # # Next we calculate the total delay
            delays = state_dict[rl_id_num][8:12]
            total_delay = sum(delays)

            # Now we compute the weighted reward
            reward = -0.5 * total_queue + -0.5 * total_wait - 0.5 * total_delay - switched + total_num_passed - 0.25 * total_em_stops

            global_queue -= total_queue
            global_wait -= total_wait
            global_delay -= total_delay
            global_switched -= switched
            global_passed += total_num_passed
            global_em -=total_em_stops
            

        return np.array([global_queue, global_wait, global_delay, global_switched, global_passed, global_em])

    def get_stats(self):
        '''
        Function to report statistics about the network at that timestep

        Returns
        -------
        - avg_velocity : the average velocity of all vehicles in the system
        - delay : the average delay of all vehicles in the system
        - near_standstill : the number of vehicles near standstill
        - energy_consumption : the consumption of all vehicles in the system
        - mpg : the miles per gallon used by the vehicles
        '''
        avg_velocity = rewards.average_velocity(self)
        delay = rewards.min_delay_unscaled(self)
        near_standstill = rewards.penalize_near_standstill(self)
        energy_consumption = rewards.energy_consumption(self)
        mpg = rewards.miles_per_gallon(self)

        return [avg_velocity, delay, near_standstill, energy_consumption, mpg]


    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}

        if self.env_params.evaluate:
            rew = -rewards.min_delay_unscaled(self)
        else:
            rew = -rewards.min_delay_unscaled(self) \
                  + rewards.penalize_near_standstill(self, gain=0.2)
        # rew = rewards.average_velocity(self)
        # each agent receives reward normalized by number of lights
        rew /= self.num_traffic_lights

        rews = {}
        for rl_id in rl_actions.keys():
            rews[rl_id] = rew
        return rews

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        for veh_ids in self.observed_ids:
            for veh_id in veh_ids:
                self.k.vehicle.set_observed(veh_id)
