import numpy as np

from .sim_class import Simulation

def pos_vehicles(data, no_warning=False):
    sim = Simulation()
    sim.data.update(data)
    sim.gen_vehicle_pos(no_warning=no_warning)
    return sim.data

def pos_targets_grid(data, grid_size=300):
    data["REALIZATIONS"] = 1
    axis = np.linspace(-data["CELL_RANGE"], data["CELL_RANGE"], grid_size)
    X, Y = np.meshgrid(axis, axis)
    target_mask = X**2 + Y**2 <= data["CELL_RANGE"]**2
    data["vehicle_x"] = X.ravel()[np.newaxis, :]
    data["vehicle_y"] = Y.ravel()[np.newaxis, :]
    data["vehicle_range"] = np.sqrt(data["vehicle_x"]**2 + data["vehicle_y"]**2)
    data["vehicle_angle"] = np.arctan2(data["vehicle_y"], data["vehicle_x"])
    n_targets = grid_size**2
    data["target_mask"] = target_mask.ravel()[np.newaxis, :]
    data["ue_mask"] = data["target_mask"]#np.zeros((1, n_targets), dtype=bool)
    data["vehicle_mask"] = data["target_mask"] | data["ue_mask"]
    return data

def pos_sensors(data, no_warning=False):
    sim = Simulation()
    sim.data.update(data)
    sim.gen_sensor_pos_random(no_warning=no_warning)
    return sim.data

def run_pre_allocation_and_assignment(data):
    sim = Simulation()
    sim.data.update(data)

    # System properties
    sim.der_sector_angle_aperture()
    sim.der_broadside_hpbw()
    sim.der_sidelobe_gain()
    sim.der_min_max_half_beamwidth()

    # Vehicle properties
    sim.der_vehicle_sector_angle()
    sim.der_vehicle_peak_beam_gain()
    sim.gen_vehicle_fading()
    sim.der_vehicle_half_beamwidth()
    sim.der_vehicle_comm_pl()

    # Sensor properties
    sim.der_sensor_angle_aperture()
    sim.der_sensor_sector_angle()
    sim.der_bs_sensor_angle()
    sim.gen_sensor_fading()
    sim.der_sensor_bs_peak_beam_gain()
    sim.der_bs_sensor_peak_beam_gain()
    sim.der_sensor_half_beamwidth()
    sim.der_sensor_comm_pl()

    # Vehicle-sensor properties
    sim.der_vehicle_sensor_mask()
    sim.der_vehicle_sensor_distance_mat()
    sim.der_vehicle_sensor_angle_mat()
    sim.der_vehicle_sensor_peak_beam_gain_mat()
    sim.der_bistatic_angle_mat()
    sim.gen_rcs_term_s()
    sim.der_vehicle_rcs_mat()
    sim.gen_vehicle_sensor_fading_mat()
    sim.der_vehicle_sensing_pl_mat()

    return sim.data

def run_angle_interference_derivations(data):
    sim = Simulation()
    sim.data.update(data)

    sim.der_interfering_vehicles_mat_angles()
    sim.der_interfering_sensors_mat_angles()

    return sim.data

def alloc_beams(data, method="orthogonal"):
    sim = Simulation()
    sim.data.update(data)
    if method == "orthogonal":
        sim.dec_beam_allocation_vehicles_full()
        sim.dec_beam_allocation_sensors_orthogonal()
    elif method == "forced_full":
        sim.dec_beam_allocation_forced_full()
    else:
        raise ValueError(f"Unknown beam allocation method: {method}")
    return sim.data

def run_beam_allocation_interference_derivations(data):
    sim = Simulation()
    sim.data.update(data)

    sim.der_interfering_vehicles_mat_beam_allocation()
    sim.der_interfering_sensors_mat_beam_allocation()

    return sim.data

def alloc_bandwidth(data, method="forced", sens_portions=None, no_interf_targets_use_full_band=False):
    sim = Simulation()
    sim.data.update(data)
    if method == "orthogonal_split_equal":
        if sens_portions is None:
            sens_portions = 0.5 # Default to equal split if not provided
        if np.array(sens_portions).ndim == 0:
            sens_portions = np.array([sens_portions])
        alloc_decisions = [(a, data["UPLINK_PRBS"]) for a in sens_portions]
        sim.dec_bandwidth_allocation_orthogonal_split_equal(alloc_decisions, no_interf_targets_use_full_band=no_interf_targets_use_full_band)
        sim.dec_sensor_bandwidth_allocation_orthogonal_split_equal([data["UPLINK_PRBS"]]*len(sens_portions))

    elif method == "forced":
        sim.dec_bandwidth_allocation(
            target=data["TARGET_SENSING_PRBS"],
            ue_sens=data["UE_SENSING_PRBS"],
            ue_comm=data["UE_COMM_PRBS"],
            uplink=data["UPLINK_PRBS"]
        )
    else:
        raise ValueError(f"Unknown bandwidth allocation method: {method}")
    return sim.data

def run_pre_assignment(data):
    sim = Simulation()
    sim.data.update(data)

    # Sensing resolution
    sim.der_range_resolution()
    sim.der_angular_resolution()
    sim.der_resolution_cell_area()
    sim.der_max_resolved_vehicles()
    sim.der_unresolution_rates()

    # Vehicle properties
    sim.der_vehicle_processing_gain()
    sim.der_vehicle_sensing_noise_power()
    sim.der_vehicle_comm_noise_power()

    # Sensor properties
    sim.der_sensor_comm_noise_power()
    sim.der_sensor_comm_snr()
    sim.der_sensor_comm_thr()

    # Sensing SNR
    sim.der_vehicle_sensing_snr_mat()

    # Sensing errors
    sim.der_vehicle_range_error_mat()
    sim.der_vehicle_angle_error_mat()
    sim.der_vehicle_position_error_mat()

    # Vehicle beamforming gain based on angle error
    sim.der_vehicle_beamforming_gain_mat()

    # UE communication SNR and throughput
    sim.der_vehicle_comm_snr_mat()
    sim.der_vehicle_side_lobe_comm_snr()
    sim.der_vehicle_comm_thr_mat()
    sim.der_vehicle_side_lobe_thr()

    # Vehicle SLAd
    sim.der_vehicle_comm_slad_mat()
    sim.der_vehicle_sensing_slad_mat()

    # Vehicle SLA satisfaction
    sim.der_side_lobe_comm_sla_satisfied()
    sim.der_comm_sla_satisfied_mat()
    sim.der_sensing_sla_satisfied_mat()

    return sim.data
    

def assign_sensors(data, method="lowest_position_error"):
    """
    Assign sensors to vehicles based on a specified method.

    Supported methods:
        - "lowest_position_error": Assign each vehicle to the sensor that would yield the lowest position error.
    """
    
    sim = Simulation()
    sim.data.update(data)

    methods = {
        "lowest_position_error": sim.dec_assign_sensor_lowest_position_error,
    }

    if method not in methods:
        raise ValueError(f"Unknown assignment method: {method}")
    
    methods[method]()

    return sim.data
            

def run_post_assignment(data, no_sensor_demand_warning=False):
    sim = Simulation()
    sim.data.update(data)

    # Remove sensor dimensions
    sim.der_apply_assignments()

    # Check if sensor throuhgput is enough to transmit the sensing information
    sim.der_sensor_demand()
    sim.der_sensor_demand_satisfied()
    sim.der_sensor_demand_satisfaction_ratio(no_sensor_demand_warning=no_sensor_demand_warning)
    sim.der_successfully_transmitted_vehicles_ratio()

    # Calculate peak throughput allocation decision for each UE
    sim.der_ue_peak_thr_alloc_decision()

    # Calculate sensing capacity
    sim.der_sensing_capacity()
    
    # Calculate SLA satisfaction and SSR
    sim.der_side_lobe_comm_ssr()
    sim.der_comm_ssr()
    sim.der_sensing_ssr()
    sim.der_ssr()

    # Calculate mean sensing and communication performance metrics
    sim.der_mean_target_position_estimation_error()
    sim.der_mean_ue_comm_thr()

    return sim.data
    