"""
Copyright (c) 2018 Ewan Barr <ebarr@mpifr-bonn.mpg.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
import logging
import uuid
import time
import datetime
import katsdptelstate
import warnings
import functools
import numpy as np
from tornado.gen import coroutine, Return, sleep, TimeoutError
from katportalclient import KATPortalClient
from katcp.resource import KATCPSensorReading
from katcp import Sensor

logger = log = logging.getLogger('mpikat.katportalclient_wrapper')


class SDPWideProductNotFound(Exception):
    pass


class CalSolutionsUnavailable(Exception):
    """Requested calibration solutions are not available from cal pipeline."""


def get_cal_inputs(view):
    """Get list of input labels associated with calibration products."""
    try:
        ants = view['antlist']
        pols = view['pol_ordering']
    except KeyError as err:
        raise CalSolutionsUnavailable(str(err))
    return [ant + pol for pol in pols for ant in ants]


def get_cal_channel_freqs(view):
    """Get sky frequencies (in Hz) associated with bandpass cal solutions."""
    try:
        bandwidth = view['bandwidth']
        center_freq = view['center_freq']
        n_chans = view['n_chans']
    except KeyError as err:
        raise CalSolutionsUnavailable(str(err))
    return center_freq + (np.arange(n_chans) - n_chans / 2) * (bandwidth / n_chans)


def _get_latest_within_interval(view, key, timeout, start_time, end_time,
                                n_parts=0):
    """Get latest value of  from telstate

    The interval is given by [, ). If
    is None the interval becomes open-ended: [, inf). In that
    case, wait for up to  seconds for a value to appear. Raise a
    :exc: or :exc: if no values were
    found in the interval.

    If  is a positive integer, the sensor is array-valued and
    split across that many parts, which are indexed by appending a sequence
    of integers to  to obtain the actual telstate keys. The values of
    the key parts will be stitched together along the first dimension of
    each array. If only some produce values within the timeout, the missing
    parts are replaced with arrays of NaNs.
    """
    # Coerce n_parts to int to catch errors early (it comes from telstate)
    n_parts = int(n_parts)
    # Handle the simple non-split case first
    if n_parts <= 0:
        if end_time is None:
            # Wait for fresh value to appear
            fresh = lambda value, ts: ts >= start_time  # noqa: E731
            view.wait_key(key, fresh, timeout)
            return view[key]
        else:
            # Assume any value in the given interval would already be there
            solution_before_end = view.get_range(key, et=end_time)
            if solution_before_end and solution_before_end[0][1] >= start_time:
                return solution_before_end[0][0]
            else:
                raise KeyError('No {} found between timestamps {} and {}'
                               .format(key, start_time, end_time))
    # Handle the split case (n_parts is now a positive integer)
    parts = []
    valid_part = None
    deadline = time.time() + timeout
    for i in range(n_parts):
        timeout_left = max(0.0, deadline - time.time())
        try:
            valid_part = _get_latest_within_interval(
                view, key + str(i), timeout_left, start_time, end_time)
        except (KeyError, katsdptelstate.TimeoutError) as err:
            if end_time is None:
                # Don't use err's msg as that will give  secs
                logger.warning('Timed out after %g seconds waiting '
                               'for telstate keys %s*', timeout, key)
            else:
                logger.warning(str(err))
            parts.append(None)
        else:
            parts.append(valid_part)
    if valid_part is None:
        raise KeyError('All {}* keys either timed out or were not found '
                       'within interval'.format(key))
    # If some (but not all) of the solution was missing, fill it with NaNs
    for i in range(n_parts):
        if parts[i] is None:
            parts[i] = np.full_like(valid_part, np.nan)
    return np.concatenate(parts)


def get_cal_solutions(view, name, timeout=0., start_time=None, end_time=None):
    """Retrieve calibration solutions from telescope state.

    Parameters
    ----------
    view : :class:
        Telstate with the appropriate view of calibration products
    name : string
        Identifier of desired calibration solutions (e.g. 'K', 'G', 'B')
    timeout : float, optional
        Time to wait for solutions to appear, in seconds
    start_time : float, optional
        Look for solutions based on data captured after this time
        (defaults to the start of time, i.e. the birth of the subarray)
    end_time : float, optional
        Look for solutions based on data captured before this time
        (defaults to the end of time)

    Returns
    -------
    solutions : dict mapping string to float / array
        Calibration solutions associated with each correlator input

    Raises
    ------
    CalSolutionsUnavailable
        If the requested cal solutions are not available for any reason
    """
    if start_time is None:
        start_time = 0.0
    # Check early whether the cal pipeline is even running
    inputs = get_cal_inputs(view)
    key = 'product_' + name
    try:
        # Bandpass-like cal is special as it has multiple parts (split cal)
        n_parts = view['product_B_parts'] if name.startswith('B') else 0
        solutions = _get_latest_within_interval(
            view, key, timeout, start_time, end_time, n_parts)
    except (KeyError, katsdptelstate.TimeoutError, ValueError) as err:
        msg = 'No {} calibration solutions found: {}'.format(name, err)
        raise CalSolutionsUnavailable(msg)
    logger.info('Found %s solutions', name)
    # The sign of katsdpcal delays are opposite to that of corr delay model
    if name.startswith('K'):
        solutions = -solutions
    # The HV delay / phase is a single number per channel per polarisation
    # for the entire array, but the solver gets estimates per antenna.
    # The cal pipeline has standardised on using the nanmedian solution
    # instead of picking the solution of the reference antenna.
    # Copy this solution for all antennas to keep the shape of the array.
    if name[1:].startswith('CROSS_DIODE'):
        with warnings.catch_warnings():
            # All antennas could have NaNs in one channel so don't warn
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            solutions[:] = np.nanmedian(solutions, axis=-1, keepdims=True)
    # Collapse the polarisation and antenna axes into one input axis at end
    solutions = solutions.reshape(solutions.shape[:-2] + (-1,))
    # Move input axis to front and pair up with input labels to form dict
    return dict(zip(inputs, np.moveaxis(solutions, -1, 0)))


def clean_bandpass(bp_gains):
    """Clean up bandpass gains by linear interpolation across flagged regions."""
    clean_gains = {}
    # Linearly interpolate across flagged regions
    for inp, bp in bp_gains.items():
        flagged = np.isnan(bp)
        if flagged.all():
            clean_gains[inp] = bp
            continue
        chans = np.arange(len(bp))
        clean_gains[inp] = np.interp(chans, chans[~flagged], bp[~flagged])
    return clean_gains


def calculate_corrections(G_gains, B_gains, delays, cal_channel_freqs,
                          target_average_correction, flatten_bandpass,
                          random_phase=False):
    """Turn cal pipeline products into corrections to be passed to F-engine."""
    average_gain = {}
    gain_corrections = {}
    # First find relative corrections per input with arbitrary global average
    for inp in G_gains:
        # Combine all calibration products for input into single array of gains
        K_gains = np.exp(-2j * np.pi * delays[inp] * cal_channel_freqs)
        gains = K_gains * B_gains[inp] * G_gains[inp]
        if np.isnan(gains).all():
            average_gain[inp] = 0.0
            gain_corrections[inp] = np.zeros(len(cal_channel_freqs), dtype='complex64')
            continue
        abs_gains = np.abs(gains)
        # Track the average gain to fix overall power level (and as diagnostic)
        average_gain[inp] = np.nanmedian(abs_gains)
        corrections = 1.0 / gains
        if not flatten_bandpass:
            # Let corrections have constant magnitude equal to 1 / (avg gain),
            # which ensures that power levels are still equalised between inputs
            corrections *= abs_gains / average_gain[inp]
        if random_phase:
            corrections *= np.exp(2j * np.pi * np.random.rand(len(corrections)))
        gain_corrections[inp] = np.nan_to_num(corrections)
    # All invalid gains (NaNs) have now been turned into zeros
    valid_average_gains = [g for g in average_gain.values() if g > 0]
    if not valid_average_gains:
        raise ValueError("All gains invalid and beamformer output will be zero!")
    global_average_gain = np.median(valid_average_gains)

    # Iterate over inputs again and fix average values of corrections
    for inp in sorted(G_gains):
        relative_gain = average_gain[inp] / global_average_gain
        if relative_gain == 0.0:
            logger.warning("%s has no valid gains and will be zeroed", inp)
            continue
        # This ensures that input at the global average gets target correction
        gain_corrections[inp] *= target_average_correction * global_average_gain
        safe_relative_gain = np.clip(relative_gain, 0.5, 2.0)
        if relative_gain == safe_relative_gain:
            logger.info("%s: average gain relative to global average = %5.2f",
                        inp, relative_gain)
        else:
            logger.warning("%s: average gain relative to global average "
                           "= %5.2f out of range, clipped to %.1f",
                           inp, relative_gain, safe_relative_gain)
            gain_corrections[inp] *= relative_gain / safe_relative_gain
    return gain_corrections


def _telstate_capture_stream(telstate, capture_block_id, stream_name):
    """Create telstate having only <stream> and <cbid_stream> views."""
    capture_stream = telstate.join(capture_block_id, stream_name)
    return telstate.view(stream_name, exclusive=True).view(capture_stream)


def get_phaseup_corrections(telstate, end_time, target_average_correction,
                            flatten_bandpass):
    """Get corrections associated with phase-up that ended at .

    Parameters
    ----------
    telstate : :class:
        Top-level telstate object
    end_time : float
        Time when phase-up successfully completed, as Unix timestamp
    target_average_correction : float
        The global average F-engine gain for all inputs to ensure good quantisation
    flatten_bandpass : bool
        True to flatten the shape of the bandpass magnitude (i.e. do different
        per-channel amplitude gains)

    Returns
    -------
    corrections : dict mapping string to array of complex
        Complex F-engine gains per channel per input that will phase up the
        inputs and optionally correct their bandpass shapes
    """
    # Obtain capture block ID associated with successful phase-up
    phaseup_cbid = telstate.get_range('sdp_capture_block_id', et=end_time)
    if not phaseup_cbid:
        # The phase-up probably happened in an earlier subarray, so report it
        all_cbids = telstate.get_range('sdp_capture_block_id', st=0)
        if not all_cbids:
            raise CalSolutionsUnavailable('Subarray has not captured any data yet')
        first_cbid, start_time = all_cbids[0]
        raise CalSolutionsUnavailable('Requested phase-up time is {} but current '
                                      'subarray only started capturing data at {} (cbid {})'
                                      .format(end_time, start_time, first_cbid))
    cbid, start_time = phaseup_cbid[0]
    view = _telstate_capture_stream(telstate, cbid, 'cal')
    _get = functools.partial(get_cal_solutions, view,
                             start_time=start_time, end_time=end_time)
    # Wait for the last relevant bfcal product from the pipeline
    try:
        hv_gains = _get('BCROSS_DIODE_SKY')
    except CalSolutionsUnavailable as err:
        logger.warning("No BCROSS_DIODE_SKY solutions found - "
                       "falling back to BCROSS_DIODE only: %s", err)
        hv_gains = _get('BCROSS_DIODE')
    hv_delays = _get('KCROSS_DIODE')
    gains = _get('G')
    bp_gains = _get('B')
    delays = _get('K')
    # Add HV delay to the usual delay
    for inp in sorted(delays):
        delays[inp] += hv_delays.get(inp, 0.0)
        if np.isnan(delays[inp]):
            logger.warning("Delay fit failed on input %s (all its "
                           "data probably flagged)", inp)
    # Add HV phase to bandpass phase
    for inp in bp_gains:
        bp_gains[inp] *= hv_gains.get(inp, 1.0)
    bp_gains = clean_bandpass(bp_gains)
    cal_channel_freqs = get_cal_channel_freqs(view)
    return calculate_corrections(gains, bp_gains, delays, cal_channel_freqs,
                                 target_average_correction, flatten_bandpass)


class KatportalClientWrapper(object):
    def __init__(self, host, callback=None):
        self._host = host
        self._client = KATPortalClient(
            host,
            on_update_callback=callback,
            logger=logging.getLogger('katcp'))

    @coroutine
    def _query(self, component, sensor, include_value_ts=False):
        log.debug("Querying sensor '{}' on component '{}'".format(
            sensor, component))
        sensor_name = yield self._client.sensor_subarray_lookup(
            component=component, sensor=sensor, return_katcp_name=False)
        log.debug("Found sensor name: {}".format(sensor_name))
        sensor_sample = yield self._client.sensor_value(
            sensor_name,
            include_value_ts=include_value_ts)
        log.debug("Sensor value: {}".format(sensor_sample))
        if sensor_sample.status != Sensor.STATUSES[Sensor.NOMINAL]:
            message = "Sensor {} not in NOMINAL state".format(sensor_name)
            log.error(message)
            raise Exception(sensor_name)
        raise Return(sensor_sample)

    @coroutine
    def get_observer_string(self, antenna):
        sensor_sample = yield self._query(antenna, "observer")
        raise Return(sensor_sample.value)

    @coroutine
    def get_observer_strings(self, antennas):
        query = "^({})_observer".format("|".join(antennas))
        log.debug("Regex query '{}'".format(query))
        sensor_samples = yield self._client.sensor_values(
            query, include_value_ts=False)
        log.debug("Sensor value: {}".format(sensor_samples))
        antennas = {}
        for key, value in sensor_samples.items():
            antennas[key.strip("_observer")] = value.value
        raise Return(antennas)

    @coroutine
    def get_antenna_feng_id_map(self, instrument_name, antennas):
        sensor_sample = yield self._query('cbf', '{}.input-labelling'.format(
            instrument_name))
        labels = eval(sensor_sample.value)
        mapping = {}
        for input_label, input_index, _, _ in labels:
            antenna_name = input_label.strip("vh").lower()
            if antenna_name.startswith("m") and antenna_name in antennas:
                mapping[antenna_name] = input_index//2
        raise Return(mapping)

    @coroutine
    def get_bandwidth(self, stream):
        sensor_sample = yield self._query('sub', 'streams.{}.bandwidth'.format(
            stream))
        raise Return(sensor_sample.value)

    @coroutine
    def get_cfreq(self, stream):
        sensor_sample = yield self._query(
            'sub', 'streams.{}.centre-frequency'.format(stream))
        raise Return(sensor_sample.value)

    @coroutine
    def get_sideband(self, stream):
        sensor_sample = yield self._query(
            'sub', 'streams.{}.sideband'.format(stream))
        raise Return(sensor_sample.value)

    @coroutine
    def get_sync_epoch(self):
        sensor_sample = yield self._query('sub', 'synchronisation-epoch')
        raise Return(sensor_sample.value)

    @coroutine
    def get_itrf_reference(self):
        sensor_sample = yield self._query('sub', 'array-position-itrf')
        x, y, z = [float(i) for i in sensor_sample.value.split(",")]
        raise Return((x, y, z))

    @coroutine
    def get_proposal_id(self):
        sensor_sample = yield self._query('sub', 'script-proposal-id')
        raise Return(sensor_sample.value)

    @coroutine
    def get_sb_id(self):
        sensor_sample = yield self._query('sub', 'script-experiment-id')
        raise Return(sensor_sample.value)

    @coroutine
    def get_telstate(self, key='_wide_'):
        sensor_sample = yield self._query('sdp', 'subarray-product-ids')
        products = sensor_sample.value.split(",")
        for product in products:
            print product
            if key in product:
                product_id = product
                break
        else:
            raise SDPWideProductNotFound
        sensor_sample = yield self._query(
            'sdp', 'spmc_{}_telstate_telstate'.format(product_id))
        raise Return(sensor_sample.value)

    @coroutine
    def get_last_phaseup_timestamp(self):
        sensor_sample = yield self._query(
            'sub', 'script-last-phaseup',
            include_value_ts=True)
        raise Return(sensor_sample.value_time)

    @coroutine
    def get_last_delay_cal_timestamp(self):
        sensor_sample = yield self._query(
            'sub', 'script-last-delay-calibration',
            include_value_ts=True)
        raise Return(sensor_sample.value_time)

    @coroutine
    def get_last_calibration_timestamp(self):
        delay_cal_error = None
        phase_cal_error = None
        try:
            delay_cal = yield self.get_last_delay_cal_timestamp()
        except Exception as error:
            delay_cal_error = error
        try:
            phase_cal = yield self.get_last_phaseup_timestamp()
        except Exception as error:
            phase_cal_error = error

        if phase_cal_error and delay_cal_error:
            raise Exception(
                "No valid calibration timestamps: delay error: {}, phaseup error: {}".format(
                    delay_cal_error, phase_cal_error))
        elif phase_cal_error:
            raise Return(delay_cal)
        elif delay_cal_error:
            raise Return(phase_cal)
        else:
            raise Return(max(delay_cal, phase_cal))

    @coroutine
    def get_gains(self):
        val = yield self.get_telstate()
        telstate_address = "{}:{}".format(*eval(val))
        last_calibration = yield self.get_last_calibration_timestamp()
        telstate = katsdptelstate.TelescopeState(telstate_address)
        corrections = get_phaseup_corrections(
            telstate,
            last_calibration, 1.0, False)
        raise Return(corrections)

    @coroutine
    def get_fbfuse_address(self):
        sensor_sample = yield self._query('fbfuse', 'fbfmc-address')
        raise Return(eval(sensor_sample.value))

    @coroutine
    def get_fbfuse_target_config(self, product_id):
        sensor_list = [
            "phase-reference",
            "coherent-beam-shape"
        ]
        fbf_config = {}
        fbfuse_proxy = yield self.get_fbfuse_proxy_id()
        prefix = "{}_fbfmc_{}_".format(fbfuse_proxy, product_id)
        query = "^{}({})$".format(
            prefix, "|".join([s.replace("-", "_") for s in sensor_list]))
        log.debug("Regex query '{}'".format(query))
        sensor_samples = yield self._client.sensor_values(
            query, include_value_ts=False)
        log.debug("Sensor value: {}".format(sensor_samples))
        for sensor_name in sensor_list:
            full_name = "{}{}".format(prefix, sensor_name.replace("-", "_"))
            sensor_sample = sensor_samples[full_name]
            log.debug(sensor_sample)
            if sensor_sample.status != Sensor.STATUSES[Sensor.NOMINAL]:
                message = "Sensor {} not in NOMINAL state".format(full_name)
                log.error(message)
                raise Exception(sensor_name)
            else:
                fbf_config[sensor_name] = sensor_sample.value
        raise Return(fbf_config)

    @coroutine
    def get_fbfuse_sb_config(self, product_id):
        sensor_list = [
            "bandwidth",
            "nchannels",
            "centre-frequency",
            "coherent-beam-count",
            "coherent-beam-count-per-group",
            "coherent-beam-ngroups",
            "coherent-beam-tscrunch",
            "coherent-beam-fscrunch",
            "coherent-beam-antennas",
            "coherent-beam-multicast-groups",
            "coherent-beam-multicast-group-mapping",
            "coherent-beam-multicast-groups-data-rate",
            "coherent-beam-heap-size",
            "coherent-beam-idx1-step",
            "coherent-beam-subband-nchans",
            "coherent-beam-time-resolution",
            "incoherent-beam-count",
            "incoherent-beam-tscrunch",
            "incoherent-beam-fscrunch",
            "incoherent-beam-antennas",
            "incoherent-beam-multicast-group",
            "incoherent-beam-multicast-group-data-rate",
            "incoherent-beam-heap-size",
            "incoherent-beam-idx1-step",
            "incoherent-beam-subband-nchans",
            "incoherent-beam-time-resolution"
        ]
        fbf_config = {}
        fbfuse_proxy = yield self.get_fbfuse_proxy_id()
        prefix = "{}_fbfmc_{}_".format(fbfuse_proxy, product_id)
        query = "^{}({})$".format(
            prefix, "|".join([s.replace("-", "_") for s in sensor_list]))
        log.debug("Regex query '{}'".format(query))
        sensor_samples = yield self._client.sensor_values(
            query, include_value_ts=False)
        log.debug("Sensor value: {}".format(sensor_samples))
        for sensor_name in sensor_list:
            full_name = "{}{}".format(prefix, sensor_name.replace("-", "_"))
            sensor_sample = sensor_samples[full_name]
            log.debug(sensor_sample)
            if sensor_sample.status != Sensor.STATUSES[Sensor.NOMINAL]:
                message = "Sensor {} not in NOMINAL state".format(full_name)
                log.error(message)
                raise Exception(sensor_name)
            else:
                fbf_config[sensor_name] = sensor_sample.value
        raise Return(fbf_config)

    @coroutine
    def get_fbfuse_coherent_beam_positions(self, product_id):
        component = product_id.replace("array", "fbfuse")
        prefix = "{}_fbfmc_{}_".format(component, product_id)
        query = "^{}.*cfbf.*$".format(prefix)
        sensor_samples = yield self._client.sensor_values(
            query, include_value_ts=False)
        beams = {}
        for key, reading in sensor_samples.items():
            beam_name = key.split("_")[-1]
            if reading.status == Sensor.STATUSES[Sensor.NOMINAL]:
                beams[beam_name] = reading.value
        raise Return(beams)

    @coroutine
    def get_fbfuse_proxy_id(self):
        sensor_sample = yield self._query('sub', 'allocations')
        for resource, _, _ in eval(sensor_sample.value):
            if resource.startswith("fbfuse"):
                raise Return(resource)
        else:
            raise Exception("No FBFUSE proxy found in current subarray")

    def get_sensor_tracker(self, component, sensor_name):
        return SensorTracker(self._host, component, sensor_name)


class Interrupt(Exception):
    pass


class SensorTracker(object):
    def __init__(self, host, component, sensor_name):
        log.debug(("Building sensor tracker activity tracker "
                   "on {} for sensor={} and component={}").format(
            host, sensor_name, component))
        self._client = KATPortalClient(
            host,
            on_update_callback=self.event_handler,
            logger=logging.getLogger('katcp'))
        self._namespace = 'namespace_' + str(uuid.uuid4())
        self._sensor_name = sensor_name
        self._component = component
        self._full_sensor_name = None
        self._state = None
        self._has_started = False

    @coroutine
    def start(self):
        if self._has_started:
            return
        log.debug("Starting sensor tracker")
        yield self._client.connect()
        result = yield self._client.subscribe(self._namespace)
        self._full_sensor_name = yield self._client.sensor_subarray_lookup(
            component=self._component, sensor=self._sensor_name,
            return_katcp_name=False)
        log.debug("Tracking sensor: {}".format(self._full_sensor_name))
        result = yield self._client.set_sampling_strategies(
            self._namespace, self._full_sensor_name,
            'event')
        sensor_sample = yield self._client.sensor_value(
            self._full_sensor_name,
            include_value_ts=False)
        self._state = sensor_sample.value
        log.debug("Initial state: {}".format(self._state))
        self._has_started = True

    @coroutine
    def stop(self):
        yield self._client.unsubscribe(self._namespace)
        yield self._client.disconnect()

    def event_handler(self, msg_dict):
        status = msg_dict['msg_data']['status']
        if status == "nominal":
            log.debug("Sensor value update: {} -> {}".format(
                self._state, msg_dict['msg_data']['value']))
            self._state = msg_dict['msg_data']['value']

    @coroutine
    def wait_until(self, state, interrupt):
        log.debug("Waiting for state='{}'".format(state))
        while True:
            if self._state == state:
                log.debug("Desired state reached")
                raise Return(self._state)
            else:
                try:
                    log.debug("Waiting on interrupt in wait_until loop")
                    yield interrupt.wait(
                        timeout=datetime.timedelta(seconds=1))
                    log.debug("Moving to next loop iteration")
                except TimeoutError:
                    continue
                else:
                    log.debug("Wait was interrupted")
                    raise Interrupt("Interrupt event was set")


class SubarrayActivity(SensorTracker):
    def __init__(self, host):
        super(SubarrayActivity, self).__init__(
            host, "subarray", "observation_activity")


if __name__ == "__main__":
    import tornado
    host = "http://portal.mkat.karoo.kat.ac.za/api/client/1"
    log = logging.getLogger('mpikat.katportalclient_wrapper')
    log.setLevel(logging.DEBUG)
    ioloop = tornado.ioloop.IOLoop.current()
    client = KatportalClientWrapper(host)

    @coroutine
    def setup():
        yield client.get_telstate()

    ioloop.run_sync(setup)
