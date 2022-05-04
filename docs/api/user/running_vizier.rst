Running Vizier
##############

An example of the entire server + client loop running locally can be found in the `unit test file`_. A manual demo can be found in the `demos folder`_.
To run the manual demo, run the following command to start the server::

  python run_vizier_server.py

which will print out an ``address`` of the form ``localhost:[PORT]``.

Then run the following command using the ``address`` to start the client::

  python run_vizier_client.py --address="localhost:[PORT]"

We explain how the core components work below.

Running the Server
------------------
To start the Vizier service, the standard way via GRPC is to do the following on the host machine::

  import grpc
  import portpicker

  # Setup Vizier Service and its data.
  servicer = vizier_server.VizierService()

  # Setup local networking.
  port = portpicker.pick_unused_port()
  address = f'localhost:{port}'

  # Setup server.
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

  vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(servicer, server)
  server.add_secure_port(address, grpc.local_server_credentials())
  server.start()

Running a Client
----------------
The user may then interact with the service via the client interface. The user first needs to setup the search space, metrics, and algorithm, in the ``StudyConfig``::

  from vizier.service import pyvizier as vz

  study_config = vz.StudyConfig() # Search space, metrics, and algorithm.
  root = study_config.search_space.select_root() # "Root" params must exist in every trial.
  root.add_float_param('learning_rate', min_value=1e-4, max_value=1e-2, scale_type=vz.ScaleType.LOG)
  root.add_int_param('num_layers', min_value=1, max_value=5)
  study_config.metric_information.append(vz.MetricInformation(name='accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE, min_value=0.0, max_value=1.0))
  study_config.algorithm = vz.Algorithm.RANDOM_SEARCH

Using the ``address`` created above in the server section, we may now create the client (e.g. on a worker machine different from the server)::

  from vizier.service import vizier_client

  client = vizier_client.create_or_load_study(
      service_endpoint=address,  # Same address as server.
      owner_id='my_name',
      client_id='my_client_id',
      study_display_name='cifar10',
      study_config=study_config)

Note that the above can be called multiple times, one on each machine, to obtain ``client_2``, ``client_3``,..., all working on the same study, especially for tuning jobs which require multiple machines to parallelize the workload.

Each client may now send requests to the server and receive responses, for example::

  client.list_trials()  # List out trials for the corresponding study.
  client.get_trial(trial_id=1)  # Get the first trial.


The default usage is to tune a user defined blackbox objective ``evaluate_trial()``, with an example shown below::

  suggestions = client.get_suggestions(suggestion_count=5)  # Batch of 5 suggestions.
  # Evaluate the suggestion(s) and report the results to Vizier.
  for trial in suggestions:
    measurement = evaluate_trial(trial)
    client.complete_trial(trial_id, measurement)

The Vizier service is designed to handle multiple concurrent clients all requesting suggestions and returning metrics.

.. _`unit test file`: https://github.com/google/vizier/blob/main/vizier/service/vizier_client_test.py
.. _`demos folder`: https://github.com/google/vizier/tree/main/demos
