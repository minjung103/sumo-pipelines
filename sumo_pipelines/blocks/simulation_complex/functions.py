# try to import LIBSUMO

try:
    import libsumo  # type: ignore  # noqa: PGH003
    LIBSUMO = True
except ImportError:
    import traci as libsumo
    LIBSUMO = False

import itertools
import polars as pl
import traci.constants as tc

from sumo_pipelines.blocks.simulation.functions import make_cmd
from sumo_pipelines.blocks.simulation_complex.config import (
    PriorityTrafficLightsRunnerConfig,
)
from sumo_pipelines.utils.nema_utils import NEMALight


class PhaseHolder:
    """
    A class that manages traffic light phases and calculates priority scores
    based on vehicle waiting times, speeds, and counts in each phase.
    """
    def __init__(
        self,
        tl,
        phase,
        sim_step,
        e2_detector_ids,
        truck_waiting_time_factor=3,
        truck_speed_factor=6,
        truck_count_factor=6,
    ):
        # Traffic light ID and phase number
        self.tl = tl
        self.phase = phase

        # E3 detector (multi-entry-exit detector)
        self._e3 = f"e3_{tl}_{phase}"
        # E2 detectors (lane area detectors)
        self._e2s = []

        # Set to track vehicle IDs currently in this phase area
        self._ids = set()
        # Dictionary to store when each vehicle entered the phase area
        self.accumulated_wtime_holder = dict()
        # Total accumulated waiting time for all vehicles in this phase
        self.accumuilated_wtime = 0
        # Speed factor calculation for priority scoring
        self.veh_speed_factor = 0
        # Total vehicle count (with truck weighting)
        self.veh_count = 0 # It's not actually used in speed metric
        # Simulation time step in seconds
        self.sim_step = sim_step

        # Weighting factors for trucks vs regular vehicles
        self.truck_waiting_time_factor = truck_waiting_time_factor
        self.truck_speed_factor = truck_speed_factor
        self.truck_count_factor = truck_count_factor

        # Initialize detector subscriptions and phase state
        self.subscribe()
        self.get_e2s(e2_detector_ids)
        self._on = True
        self.turn_off()  # Start with phase turned off

    @property
    def on(self):
        """Returns whether this phase is currently active"""
        return self._on

    def get_e2s(self, all_e2_detector_ids):
        self._e2s = list(
            filter(lambda x: f"{self.tl}_{self.phase}" in x, all_e2_detector_ids)
        )

    def turn_off(self):
        """
        Deactivate this phase by setting E2 detector vehicle counts to 0
        This effectively tells the traffic light controller that no vehicles
        are waiting in this phase
        """
        if self._on:
            # Override each E2 detector to report 0 vehicles
            for e2 in self._e2s:
                libsumo.lanearea.overrideVehicleNumber(e2, 0)
        self._on = False

    def turn_on(self):
        """
        Activate this phase by setting E2 detector vehicle counts to 1
        This signals to the traffic light controller that vehicles are waiting
        """
        # Set each E2 detector to report 1 vehicle (indicating demand)
        for e2 in self._e2s:
            libsumo.lanearea.overrideVehicleNumber(e2, 1)
        self._on = True

    def subscribe(self):
        """
        Subscribe to the E3 detector to receive real-time vehicle ID updates
        E3 detectors track vehicles entering and exiting specific areas
        """
        libsumo.multientryexit.subscribe(
            self._e3,
            [
                tc.LAST_STEP_VEHICLE_ID_LIST,
            ],
        )

    def update(self, e3_subs, veh_subs, sim_time):
        """
        Update vehicle tracking and calculate priority metrics for this phase
        
        Args:
            e3_subs: E3 detector subscription results (vehicle IDs in areas)
            veh_subs: Vehicle subscription results (speed, class, etc.)
            sim_time: Current simulation time in seconds
        """
        # Get current vehicle IDs and their truck status from E3 detector
        ids = set()
        for _id in e3_subs[self._e3][tc.LAST_STEP_VEHICLE_ID_LIST]:
            if _id in veh_subs:
                # Check if vehicle is a truck by looking for 't' in vehicle class
                is_truck = "t" in veh_subs[_id][tc.VAR_VEHICLECLASS]
                ids.add((_id, is_truck))
            else:
                print(f"Vehicle {_id} not found in vehicle subscriptions. This should not happen.")

        # Calculate which vehicles entered and left this phase area
        add_ids = ids.difference(self._ids)  # New vehicles entering
        remove_ids = self._ids.difference(ids)  # Vehicles that left

        # Update vehicle tracking
        self._ids = ids
        
        # Record entry time for new vehicles
        for veh_id in add_ids:
            self.accumulated_wtime_holder[veh_id] = sim_time

        # Remove tracking for vehicles that left
        for veh_id in remove_ids:
            self.accumulated_wtime_holder.pop(veh_id)

        # Reset counters
        self.veh_count = 0 # It's not actually used in speed metric
        self.veh_speed_factor = 0
        self.accumulated_wtime = 0
        
        # Consider truck priority
        for _id, truck in self._ids:
            # Vehicle count with truck weighting (trucks count more)
            self.veh_count += self.truck_count_factor * truck + 1 # It's not actually used in speed metric
            
            # Speed factor with truck weighting (truck speeds matter more)
            current_speed = veh_subs[_id][tc.VAR_SPEED]
            self.veh_speed_factor += max(
                current_speed * (self.truck_speed_factor * truck + 1), 0
            )

            # Waiting time calculation with truck weighting
            waiting_time = sim_time - self.accumulated_wtime_holder[(_id, truck)]
            self.accumulated_wtime += max(
                waiting_time * (self.truck_waiting_time_factor * truck + 1),
                0,
            )


# Vehicle subscription parameters - what data to collect for each vehicle
_vehicle_subscriptions = (
    tc.VAR_VEHICLECLASS,     # Vehicle type (car, truck, etc.)
    tc.VAR_SPEED,            # Current speed
    tc.VAR_POSITION,         # X,Y coordinates
    tc.VAR_FUELCONSUMPTION,  # Fuel consumption rate
    tc.VAR_ACCELERATION,     # Current acceleration
    tc.VAR_LANE_ID,          # Current lane
    tc.VAR_EMISSIONCLASS,    # Emission class
    tc.VAR_TIMELOSS,         # Time lost due to traffic
)

def traci_priority_light_control(
    config: PriorityTrafficLightsRunnerConfig, *args, **kwargs
) -> None:
    """
    Main function that runs the adaptive traffic light control simulation
    using priority-based phase selection algorithm
    """
 
    sumo_cmd = make_cmd(config=config)

    # Extract weight parameters
    # Mainline weights: for major through movements (phases 2,6 and similar)
    mainline_weights = (
        config.intersection_weights.mainline_a,  # Speed component weight (alpha_main)
        config.intersection_weights.mainline_b,  # Waiting time component weight (beta_main)
        config.intersection_weights.mainline_c,  # Base priority weight (gamma_main)
        config.intersection_weights.mainline_d,  # Don't use this factor
        config.intersection_weights.mainline_e,  # Don't use this factor
    )
    
    # Side street weights: for minor movements (left turns, side streets)
    side_weights = (
        config.intersection_weights.side_a,      # Speed component weight (alpha_branch)
        config.intersection_weights.side_b,      # Waiting time component weight (beta_branch)
        config.intersection_weights.side_c,      # Base priority weight (gamma_branch)
        config.intersection_weights.side_d,      # Don't use this factor
    )

    # Open output file for simulation command logging if specified
    if config.simulation_output:
        f = open(config.simulation_output, "w")
        f.write(" ".join(sumo_cmd))

    # Start SUMO simulation
    libsumo.start(sumo_cmd, stdout=f)

    # Run warmup period to stabilize traffic conditions
    libsumo.simulation.step(config.warmup_time)
    
    # Get simulation time step in milliseconds
    step_size = int(libsumo.simulation.getDeltaT() * 1000)

    # Get all E2 (lane area) detectors in the simulation
    e2_detectors = libsumo.lanearea.getIDList()

    # Initialize traffic light controllers
    lights = {}
    for junction, programID, file in config.controlled_intersections:
        # Load NEMA traffic light configuration from XML
        nema_light = NEMALight.from_xml(xml=file, id=junction, programID=programID)
        
        # Get valid phase combinations (e.g., [(2,6), (1,5), (3,7), (4,8)])
        valid_combos = [
            combo if combo[0] != combo[1] else (combo[0],)
            for combo in nema_light.get_valid_phase_combos()
        ]
        
        # Create PhaseHolder objects for each individual phase
        phase_holders = {
            p.name: PhaseHolder(
                junction,
                p.name,
                step_size / 1000,  # Convert to seconds
                e2_detectors,
                truck_waiting_time_factor=config.intersection_weights.truck_waiting_time_factor,
                truck_speed_factor=config.intersection_weights.truck_speed_factor,
            )
            for p in nema_light.get_phase_list()
        }
        
        # Store both valid combinations and phase holders for this intersection
        lights[junction] = (valid_combos, phase_holders)

        # Set up traffic light control and subscriptions
        libsumo.trafficlight.setProgram(junction, programID)
        libsumo.trafficlight.subscribe(
            junction, [tc.TL_RED_YELLOW_GREEN_STATE, tc.VAR_NAME]
        )

    # Initialize simulation timing variables
    sim_time = int(libsumo.simulation.getTime() * 1000)  # Current time in ms
    end_time = int(config.end_time * 1000)               # End time in ms
    action_step = config.action_step * step_size         # Control decision interval

    # Subscribe to all existing vehicles at simulation start
    for veh_id in libsumo.vehicle.getIDList():
        libsumo.vehicle.subscribe(veh_id, _vehicle_subscriptions)

    # Data collection
    fuel_vec = []
    veh_vec = []

    all_phases = list(itertools.product(*(p[0] for p in lights.values()))) # I don't know what is
    tl_index = list(lights.keys())  # Intersection Index

    # Main simulation loop
    while sim_time < end_time:
        # Advance simulation by one time step
        libsumo.simulation.step()

        # Get current detector and vehicle data
        e3_subs = libsumo.multientryexit.getAllSubscriptionResults()
        veh_subs = libsumo.vehicle.getAllSubscriptionResults()
        signal_subs = libsumo.trafficlight.getAllSubscriptionResults()

        # Collect vehicle data for analysis
        for veh, veh_info in veh_subs.items():
            fuel_vec.append([
                veh,                                    # Vehicle ID
                sim_time / 1000,                       # Time in seconds
                veh_info[tc.VAR_SPEED],                # Speed
                veh_info[tc.VAR_ACCELERATION],         # Acceleration
                *veh_info[tc.VAR_POSITION],            # X, Y position
                veh_info[tc.VAR_FUELCONSUMPTION],      # Fuel consumption
                veh_info[tc.VAR_LANE_ID],              # Lane ID
                veh_info[tc.VAR_EMISSIONCLASS],        # Emission class
                veh_info[tc.VAR_TIMELOSS],             # Time loss
            ])
            veh_vec.append(veh)

        # Update all phase holders with current vehicle information
        for tl_id, (_, phase_holders) in lights.items():
            for phase in phase_holders.values():
                phase.update(e3_subs, veh_subs, sim_time / 1000)

        # Make control decisions at specified intervals
        if sim_time % action_step == 0:
            combo_scores = []

            # Calculate priority scores for each intersection
            for tl, (phase_combos, phase_holders) in lights.items():
                combo_scores.append({})
                
                # Evaluate each possible phase combination for this intersection
                for combo in phase_combos:
                    # Determine if this is a mainline or side street movement
                    if combo == (2, 6):  # Major through movements
                        weights = mainline_weights
                    else:  # Side street and turning movements
                        weights = side_weights

                    combo_score = 0

                    # Calculate score for each phase in the combination
                    for phase in combo:
                        # Priority scoring components:
                        speed_component = weights[0] * phase_holders[phase].veh_speed_factor
                        wait_component = weights[1] * phase_holders[phase].accumulated_wtime * 60
                        base_component = weights[2] * 60
                        
                        phase_total = speed_component + wait_component + base_component
                        combo_score += phase_total
                    
                    # Store total score for this combination
                    combo_scores[-1][combo] = combo_score

            # Select best phase combination for each intersection independently
            best_combo = []
            for tl_i in range(len(tl_index)):
                intersection_scores = combo_scores[tl_i]
                # Choose combination with highest priority score
                best_intersection_combo = max(intersection_scores, key=intersection_scores.get)
                best_combo.append(best_intersection_combo)

            best_combo = tuple(best_combo)

            # Implement selected phase combinations
            for i, (tl_id, (_, phase_holders)) in enumerate(lights.items()):
                selected_combo = best_combo[i]
                
                # Activate/deactivate phases based on selection
                for p_num, phase in phase_holders.items():
                    if p_num in selected_combo:
                        # Side Road Protection Logic: 
                        # When a side road phase gets activated for the first time,
                        # artificially reduce waiting times of vehicles already in the detector area.
                        # This ensures that once side road gets green signal, ALL vehicles 
                        # that were waiting can clear through before main road regains priority.
                        # Without this protection, main road (with higher vehicle volume) 
                        # would quickly regain priority after just 1-2 side road vehicles pass,
                        # leaving remaining side road vehicles to wait much longer.
                        if p_num != 2 and p_num != 6:
                            # Artificially reduce waiting time to give these vehicles 
                            # sustained priority until they clear the intersection
                            for veh_id in list(phase._ids):
                                if veh_id in phase.accumulated_wtime_holder:
                                    phase.accumulated_wtime_holder[veh_id] -= 1000000

                        phase.turn_on()   # Activate this phase
                    else:
                        phase.turn_off()  # Deactivate this phase

        # Subscribe to newly departed vehicles
        for veh_id in libsumo.simulation.getDepartedIDList():
            libsumo.vehicle.subscribe(veh_id, _vehicle_subscriptions)

        # Advance simulation time
        sim_time += step_size

    # Close simulation
    libsumo.close()

    # Save collected data to Parquet file for analysis
    df = pl.DataFrame(
        fuel_vec,
        schema={
            "id": pl.Utf8,           # Vehicle ID
            "time": pl.Float64,      # Simulation time
            "speed": pl.Float64,     # Vehicle speed
            "accel": pl.Float64,     # Vehicle acceleration
            "x": pl.Float64,         # X coordinate
            "y": pl.Float64,         # Y coordinate
            "fuel": pl.Float64,      # Fuel consumption
            "lane": pl.Utf8,         # Lane ID
            "eclass": pl.Utf8,       # Emission class
            "time_loss": pl.Float64, # Time lost in traffic
        },
    )

    # Write results to file
    df.write_parquet(config.fcd_output)
