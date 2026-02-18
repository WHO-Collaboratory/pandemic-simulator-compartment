GRAPHQL_QUERY = """query GetSimulationJobById($id: ID!) {
  getSimulationJob(id: $id) {
    id
    admin_unit_0_id
    admin_unit_1_id
    admin_unit_2_id
    createdAt
    disease_id
    end_date
    owner
    selected_infected_population
    selected_population
    simulation_name
    simulation_type
    run_mode
    start_date
    tag_id
    time_steps
    travel_volume {
      leaving
      returning
    }
    updatedAt
    AdminUnit0 {
      id
      center_lat
    }
    AdminUnit1 {
      id
      center_lat
    }
    AdminUnit2 {
      id
      center_lat
    }
    Disease {
      id
      createdAt
      disease_name
      disease_type
	disease_nodes {
        type
        data {
          alias
          label
        }
        id
      }
      immunity_period
      interactions_per_period
      intervention_nodes {
        data {
          adherence_max
          adherence_min
          alias
          end_date
          end_threshold
          end_threshold_node_id
          label
          start_date
          start_threshold
          start_threshold_node_id
        }
        id
        type
      }
      model_type
      transmission_edges {
        data {
          transmission_rate
          variance_params {
            has_variance
            distribution_type
            field_name
            min
            max
          }
        }
        id
        source
        target
        type
      }
      updatedAt
    }
    interventions {
      adherence_max
      adherence_min
      end_date
      end_threshold
      end_threshold_node_id
      id
      label
      start_date
      start_threshold
      start_threshold_node_id
      type
      transmission_percentage
      hour_reduction
      variance_params {
        has_variance
        distribution_type
        field_name
        min
        max
      }
    }
    Interventions {
      items {
        id
        intervention_id
        Intervention {
          id
          name
          display_name
        }
        adherence_min
        adherence_max
        transmission_percentage
        start_date
        end_date
        start_threshold
        end_threshold
        start_threshold_node_id
        end_threshold_node_id
        hour_reduction
        FieldConfigs {
          items {
            id
            field_key
            has_variance
            distribution_type
            min
            max
          }
        }
      }
    }
    case_file {
      admin_zones {
        id
        admin_code
        admin_iso_code
        admin_level
        center_lat
        center_lon
        viz_name
        name
        population
        osm_id
        infected_population
        seroprevalence
        temp_min
        temp_max
        temp_mean
      }
      demographics {
        age_0_17
        age_18_55
        age_56_plus
      }
    }
  }
}"""

ADMIN_UNITS_BY_SIM_JOB_QUERY = """query SimulationJobAdminUnitsBySimulationJobId(
  $simulation_job_id: ID!
  $limit: Int
  $nextToken: String
) {
  simulationJobAdminUnitsBySimulationJobId(
    simulation_job_id: $simulation_job_id
    limit: $limit
    nextToken: $nextToken
  ) {
    items {
      id
      admin_unit_id
      name
      population
      infected_population
      seroprevalence
      vector_population
      temp_min
      temp_max
      temp_mean
    }
    nextToken
  }
}"""

SEARCH_ADMIN_UNITS_QUERY = """query SearchAdminUnits(
  $filter: SearchableAdminUnitFilterInput
  $limit: Int
) {
  searchAdminUnits(filter: $filter, limit: $limit) {
    items {
      id
      admin_code
      admin_iso_code
      admin_level
      center_lat
      center_lon
      viz_name
      name
      osm_id
    }
  }
}"""