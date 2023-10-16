"""Multi-agent traffic light Environment Initializer"""


from multi_agent_traffic_grid import MultiTrafficLightGridPOEnv
from flow.envs.traffic_light_grid import ADDITIONAL_ENV_PARAMS
from flow.networks import TrafficLightGridNetwork
from traffic_light_grid import CustomTrafficLightGridNetwork0, CustomTrafficLightGridNetwork1 
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
import numpy as np



class TrafficGridEnv:
    '''
    Class to create a traffic grid environment
    '''

    def __init__(self, N_ROWS, N_COLUMNS, speed = 35, render=False, flow = None, eval_dict = None, net_name=None, speed_dev=0.1):

        # Experiment parameters
        # N_ROLLOUTS = 10  # number of rollouts per training iteration
        # N_CPUS = 25  # number of parallel workers

        # Environment parameters
        HORIZON = 400  # time horizon of a single rollout
        V_ENTER = speed # enter speed for departing vehicles
        INNER_LENGTH = 400  # length of inner edges in the traffic light grid network
        LONG_LENGTH = 400# length of final edge in route
        SHORT_LENGTH = 400  # length of edges that vehicles start on
        # number of vehicles originating in the left, right, top, and bottom edges
        # N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 20, 20, 20, 20
        N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1
        EDGE_INFLOW = flow # inflow rate of vehicles at every edge
        if eval_dict is not None:
            EDGE_INFLOW = eval_dict["flow"]
        

        # we place a sufficient number of vehicles to ensure they confirm with the
        # total number specified above. We also use a "right_of_way" speed mode to
        # support traffic light compliance
        vehicles = VehicleParams()
        num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
        vehicles.add(
            veh_id="human",
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=2.5,
                max_speed=V_ENTER,
                # decel=2.5,  # avoid collisions at emergency stops
                # accel=1,
                speed_dev=speed_dev,
                speed_mode="right_of_way",
            ),
            routing_controller=(GridRouter, {}),
            num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS)

        # inflows of vehicles are place on all outer edges (listed here)
        outer_edges = [] 
        outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
        outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
        outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
        outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]
        
        # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
        inflow = InFlows()
        for i, edge in enumerate(outer_edges):
            inflow.add(
                veh_type="human",
                edge=edge,
                vehs_per_hour=EDGE_INFLOW[i],
                depart_lane="free",
                depart_speed=V_ENTER)

        network = TrafficLightGridNetwork
        if net_name == "Custom0":
            network = CustomTrafficLightGridNetwork0
        elif net_name == "Custom1":
            network = CustomTrafficLightGridNetwork1
        self.flow_params = dict(
            # name of the experiment
            exp_tag="grid_0_{}x{}_i_multiagent".format(N_ROWS, N_COLUMNS),

            # name of the flow environment the experiment is running on
            env_name=MultiTrafficLightGridPOEnv,

            # name of the network class the experiment is running on
            network=network,

            # simulator that is used by the experiment
            simulator='traci',

            # sumo-related parameters (see flow.core.params.SumoParams)
            sim=SumoParams(
                restart_instance=True,
                sim_step=1,
                render=render,
            ),

            # environment related parameters (see flow.core.params.EnvParams)
            env=EnvParams(
                horizon=HORIZON,
                additional_params={
                    "target_velocity": V_ENTER + 5,
                    "switch_time": 1,
                    "num_observed": 50,
                    "discrete": False,
                    "tl_type": "controlled",
                    "num_local_edges": 4,
                    "num_local_lights": 4,
                },
            ),

            # network-related parameters (see flow.core.params.NetParams and the
            # network's documentation or ADDITIONAL_NET_PARAMS component)
            net=NetParams(
                inflows=inflow,
                additional_params={
                    "speed_limit": 35,  # inherited from grid0 benchmark
                    "grid_array": {
                        "short_length": SHORT_LENGTH,
                        "inner_length": INNER_LENGTH,
                        "long_length": LONG_LENGTH,
                        "row_num": N_ROWS,
                        "col_num": N_COLUMNS,
                        "cars_left": N_LEFT,
                        "cars_right": N_RIGHT,
                        "cars_top": N_TOP,
                        "cars_bot": N_BOTTOM,
                    },
                    "horizontal_lanes": 1,
                    "vertical_lanes": 1,
                },
            ),

            # vehicles to be placed in the network at the start of a rollout (see
            # flow.core.params.VehicleParams)
            veh=vehicles,

            # parameters specifying the positioning of vehicles upon initialization
            # or reset (see flow.core.params.InitialConfig)
            initial=InitialConfig(
                spacing='custom',
                shuffle=True,
            ),
        )

    def make_env(self):

        create_env, env_name = make_create_env(params=self.flow_params, version=0)

        # Register as rllib env
        # register_env("alpha", create_env)

        env = create_env()
        return env

class TrafficAction():
    def __init__(self, act, N_ROWS, N_COLUMNS):
        self.discrete = False
        self.actions = {"center{}".format(i): act for i in range(N_ROWS * N_COLUMNS)}

    def __getitem__(self, key):
        actions = self.items()
        for k, a in actions:
            if k == key:
                return a
        return a

    def __setitem__(self, key, newvalue):
        self.actions[key] = newvalue

    def keys(self):
        return self.actions.keys()

    def items(self):
        return zip(self.actions.keys(), self.actions.values())

    def set_action(self, act_tensor):
        '''
        Function to set next action from a tensor
        Attributes
        ----------
        - act_tensor : a tensor with the action [num_lights]
        '''
        bool_tensor = act_tensor >= 0.5
        int_tensor = bool_tensor.int()
        for i in range(len(act_tensor)):
            self.__setitem__("center{}".format(i), int(int_tensor[i]))