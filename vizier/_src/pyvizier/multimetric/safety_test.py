from vizier import pyvizier
from vizier._src.pyvizier.multimetric import safety
from absl.testing import absltest


class SafeAcquisitionTest(absltest.TestCase):

  def testSafeMeasurements(self):
    info1 = pyvizier.MetricInformation(
        name='safety1',
        goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE,
        safety_threshold=0.1)
    info2 = pyvizier.MetricInformation(
        name='safety2',
        goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
        safety_threshold=0.5)
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1, info2]))
    safe_measurement = pyvizier.Measurement(metrics={
        'safety1': pyvizier.Metric(value=3.1),
        'safety2': pyvizier.Metric(value=0.3)
    })
    unsafe1_measurement = pyvizier.Measurement(metrics={
        'safety1': pyvizier.Metric(value=-3.1),
        'safety2': pyvizier.Metric(value=0.4)
    })
    unsafe2_measurement = pyvizier.Measurement(metrics={
        'safety1': pyvizier.Metric(value=3.1),
        'safety2': pyvizier.Metric(value=0.7)
    })
    no_safety_measurement = pyvizier.Measurement(
        metrics={'safety1': pyvizier.Metric(value=3.1)})

    self.assertListEqual(
        checker.are_measurements_safe([
            safe_measurement, unsafe1_measurement, unsafe2_measurement,
            no_safety_measurement
        ]), [True, False, False, True])

  def testSafeTrials(self):
    info1 = pyvizier.MetricInformation(
        name='obj', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    info2 = pyvizier.MetricInformation(
        name='safety',
        goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
        safety_threshold=0.5)
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1, info2]))

    safe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.3
        }))
    unsafe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.7
        }))
    no_safety_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={'obj': 0.2}))

    # Empty Trial is assumed to be safe.
    self.assertListEqual(
        checker.are_trials_safe(
            [safe_trial, unsafe_trial, no_safety_trial,
             pyvizier.Trial()]), [True, False, True, True])

  def testNoSafety(self):
    info1 = pyvizier.MetricInformation(
        name='obj', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1]))

    safe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.3
        }))
    unsafe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.7
        }))

    # With no safety metrics, all Trials are safe..
    self.assertListEqual(
        checker.are_trials_safe([safe_trial, unsafe_trial,
                                 pyvizier.Trial()]), [True, True, True])


if __name__ == '__main__':
  absltest.main()
