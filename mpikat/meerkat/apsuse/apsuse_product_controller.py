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

import logging
import time
import os
import json
from copy import deepcopy
from tornado.gen import coroutine, Return, sleep
from tornado.locks import Event
from katcp import Sensor, Message
from mpikat.core.worker_pool import WorkerAllocationError
from mpikat.core.utils import LoggingSensor
from mpikat.meerkat.katportalclient_wrapper import Interrupt
from mpikat.meerkat.apsuse.apsuse_config import ApsConfigGenerator

log = logging.getLogger("mpikat.apsuse_product_controller")

DEFAULT_FILE_LENGTH = 300.0 #seconds


class ApsProductStateError(Exception):
    def __init__(self, expected_states, current_state):
        message = "Possible states for this operation are '{}', but current state is '{}'".format(
            expected_states, current_state)
        super(ApsProductStateError, self).__init__(message)


class ApsProductController(object):
    """
    Wrapper class for an APSUSE product.
    """
    STATES = ["idle", "preparing", "ready", "starting", "capturing", "stopping", "error"]
    IDLE, PREPARING, READY, STARTING, CAPTURING, STOPPING, ERROR = STATES

    def __init__(self, parent, product_id, katportal_client, proxy_name):
        """
        @brief      Construct new instance

        @param      parent            The parent ApsMasterController instance

        @param      product_id        The name of the product

        @param      katportal_client       An katportal client wrapper instance

        @param      proxy_name        The name of the proxy associated with this subarray (used as a sensor prefix)

        #NEED FENG CONFIG

        @param      servers           A list of ApsWorkerServer instances allocated to this product controller
        """
        self.log = logging.getLogger(
            "mpikat.apsuse_product_controller.{}".format(product_id))
        self.log.debug(
            "Creating new ApsProductController with args: {}".format(
                ", ".join([str(i) for i in (
                parent, product_id, katportal_client, proxy_name)])))
        self._parent = parent
        self._product_id = product_id
        self._katportal_client = katportal_client
        self._proxy_name = proxy_name
        self._managed_sensors = []
        self._worker_config_map = {}
        self._servers = []
        self._fbf_sb_config = None
        self._state_interrupt = Event()
        self._base_output_dir = "/output/"
        self._coherent_beam_tracker = None
        self._incoherent_beam_tracker = None
        self.setup_sensors()

    def __del__(self):
        self.teardown_sensors()

    def info(self):
        """
        @brief    Return a metadata dictionary describing this product controller
        """
        out = {
            "state":self.state,
            "proxy_name":self._proxy_name
        }
        return out

    def add_sensor(self, sensor):
        """
        @brief    Add a sensor to the parent object

        @note     This method is used to wrap calls to the add_sensor method
                  on the parent ApsMasterController instance. In order to
                  disambiguate between sensors from describing different products
                  the associated proxy name is used as sensor prefix. For example
                  the "servers" sensor will be seen by clients connected to the
                  ApsMasterController server as "<proxy_name>-servers" (e.g.
                  "apsuse_1-servers").
        """
        prefix = "{}.".format(self._product_id)
        if sensor.name.startswith(prefix):
            self._parent.add_sensor(sensor)
        else:
            sensor.name = "{}{}".format(prefix, sensor.name)
            self._parent.add_sensor(sensor)
        self._managed_sensors.append(sensor)

    def setup_sensors(self):
        """
        @brief    Setup the default KATCP sensors.

        @note     As this call is made only upon an APSUSE configure call a mass inform
                  is required to let connected clients know that the proxy interface has
                  changed.
        """
        self._state_sensor = LoggingSensor.discrete(
            "state",
            description="Denotes the state of this APS instance",
            params=self.STATES,
            default=self.IDLE,
            initial_status=Sensor.NOMINAL)
        self._state_sensor.set_logger(self.log)
        self.add_sensor(self._state_sensor)

        self._fbf_sb_config_sensor = Sensor.string(
            "fbfuse-sb-config",
            description="The full FBFUSE schedule block configuration",
            default="",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._fbf_sb_config_sensor)

        self._worker_configs_sensor = Sensor.string(
            "worker-configs",
            description="The configurations for each worker server",
            default="",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._worker_configs_sensor)

        self._servers_sensor = Sensor.string(
            "servers",
            description="The worker server instances currently allocated to this product",
            default=",".join(["{s.hostname}:{s.port}".format(s=server) for server in self._servers]),
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._servers_sensor)

        self._data_rate_per_worker_sensor = Sensor.float(
            "data-rate-per-worker",
            description="The maximum ingest rate per APSUSE worker server",
            default=20000000000.0,
            unit="bits/s",
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._data_rate_per_worker_sensor)

        self._current_recording_directory_sensor = Sensor.string(
            "current-recording-directory",
            description="The current directory for recording from this subarray",
            default="",
            initial_status=Sensor.UNKNOWN
            )
        self.add_sensor(self._current_recording_directory_sensor)

        self._current_recording_sensor = Sensor.string(
            "recording-params",
            description="The parameters of the current APSUSE recording",
            default="",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._current_recording_sensor)

        self._parent.mass_inform(Message.inform('interface-changed'))
        self._state_sensor.set_value(self.READY)

    def teardown_sensors(self):
        """
        @brief    Remove all sensors created by this product from the parent server.

        @note     This method is required for cleanup to stop the APS sensor pool
                  becoming swamped with unused sensors.
        """
        for sensor in self._managed_sensors:
            self._parent.remove_sensor(sensor)
        self._managed_sensors = []
        self._parent.mass_inform(Message.inform('interface-changed'))

    @property
    def servers(self):
        return self._servers

    @property
    def capturing(self):
        return self.state == self.CAPTURING

    @property
    def idle(self):
        return self.state == self.IDLE

    @property
    def starting(self):
        return self.state == self.STARTING

    @property
    def stopping(self):
        return self.state == self.STOPPING

    @property
    def ready(self):
        return self.state == self.READY

    @property
    def preparing(self):
        return self.state == self.PREPARING

    @property
    def error(self):
        return self.state == self.ERROR

    @property
    def state(self):
        return self._state_sensor.value()

    def set_error_state(self, message):
        self._state_sensor.set_value(self.ERROR)

    @coroutine
    def configure(self):
        pass

    @coroutine
    def deconfigure(self):
        pass

    @coroutine
    def disable_all_writers(self):
        self.log.debug("Disabling all writers")
        for server in self._servers:
            try:
                yield server.disable_writers()
            except Exception as error:
                self.log.exception("Failed to disable writers on {}: {}".format(
                    server, str(error)))

    def set_data_rate_per_worker(self, value):
        if (value < 1e9) or (value > 25e9):
            log.warning("Suspect data rate set for workers: {} bits/s".format(
                value))
        self._data_rate_per_worker_sensor.set_value(value)

    @coroutine
    def enable_writers(self):
        self.log.info("Enabling writers")
        self.log.debug("Getting beam positions")
        beam_map = yield self._katportal_client.get_fbfuse_coherent_beam_positions(self._product_id)
        target_config = yield self._katportal_client.get_fbfuse_target_config(self._product_id)
        beam_map.update({"ifbf00000": target_config["phase-reference"]})
        self.log.debug("Beam map: {}".format(beam_map))
        coherent_tsamp = self._fbf_sb_config["coherent-beam-tscrunch"] * self._fbf_sb_config["nchannels"] / self._fbf_sb_config["bandwidth"]
        incoherent_tsamp = self._fbf_sb_config["incoherent-beam-tscrunch"] * self._fbf_sb_config["nchannels"] / self._fbf_sb_config["bandwidth"]

        # Now get all information required for APSMETA file
        output_dir = "{}/{}".format(
            self._base_output_dir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir)
        proposal_id = yield self._katportal_client.get_proposal_id()
        sb_id = yield self._katportal_client.get_sb_id()
        apsuse_meta = {
            "centre_frequency": self._fbf_sb_config["centre-frequency"],
            "bandwidth": self._fbf_sb_config["bandwidth"],
            "coherent_nchans": self._fbf_sb_config["nchannels"] / self._fbf_sb_config["coherent-beam-fscrunch"],
            "coherent_tsamp": coherent_tsamp,
            "incoherent_nchans": self._fbf_sb_config["nchannels"] / self._fbf_sb_config["incoherent-beam-fscrunch"],
            "incoherent_tsamp": incoherent_tsamp,
            "project_name": proposal_id,
            "sb_id": sb_id,
            "utc_start": time.strftime("%Y/%m/%d %H:%M:%S"),
            "output_dir": output_dir.replace("/DATA/", "/beegfs/DATA/TRAPUM/"),
            "beamshape": target_config["coherent-beam-shape"],
            "boresight": target_config["phase-reference"],
            "beams": beam_map
        }

        # Generate user friendly formatting for the current recording:
        format_mapping = (
            ("Centre frequency:", "centre_frequency", "Hz"),
            ("Bandwidth:", "bandwidth", "Hz"),
            ("CB Nchannels:", "coherent_nchans", ""),
            ("CB sampling:", "coherent_tsamp", "s"),
            ("IB Nchannels:", "incoherent_nchans", ""),
            ("IB sampling:", "incoherent_tsamp", "s"),
            ("Project ID:", "project_name", ""),
            ("SB ID:", "sb_id", ""),
            ("UTC start:", "utc_start", ""),
            ("Directory:", "output_dir", "")
            )
        formatted_apsuse_meta = "<br />".join(("<font color='lightblue'><b>{}</b></font> {} {}".format(name,
            apsuse_meta[key], unit) for name, key, unit in format_mapping))
        formatted_apsuse_meta = "<p>{}</p>".format(formatted_apsuse_meta)
        self._current_recording_sensor.set_value(formatted_apsuse_meta)
        self._current_recording_directory_sensor.set_value(output_dir)
        try:
            with open("{}/apsuse.meta".format(output_dir), "w") as f:
                f.write(json.dumps(apsuse_meta))
        except Exception:
            log.exception("Could not write apsuse.meta file")

        enable_futures = []
        for server in self._servers:
            worker_config = self._worker_config_map[server]
            sub_beam_list = {}
            for beam in worker_config.incoherent_beams():
                if beam in beam_map:
                    sub_beam_list[beam] = beam_map[beam]
            for beam in worker_config.coherent_beams():
                if beam in beam_map:
                    sub_beam_list[beam] = beam_map[beam]
            enable_futures.append(server.enable_writers(sub_beam_list, output_dir))

        for ii, future in enumerate(enable_futures):
            try:
                yield future
            except Exception as error:
                self.log.exception("Failed to enable writers on server {}: {}".format(
                    self._servers[ii], str(error)))

    @coroutine
    def capture_start(self):
        if not self.ready:
            raise ApsProductStateError([self.READY], self.state)
        self._state_sensor.set_value(self.STARTING)
        self.log.debug("Product moved to 'starting' state")
        # At this point assume we do not know about the SB config and get everything fresh
        proposal_id = yield self._katportal_client.get_proposal_id()
        sb_id = yield self._katportal_client.get_sb_id()
        # determine base output path
        # /output/{proposal_id}/{sb_id}/
        # scan number will be added to the path later
        # The /DATA/ path is usually a mount of /beegfs/DATA/TRAPUM
        self._base_output_dir = "/DATA/{}/{}/".format(proposal_id, sb_id)
        self._fbf_sb_config = yield self._katportal_client.get_fbfuse_sb_config(self._product_id)
        self._fbf_sb_config_sensor.set_value(self._fbf_sb_config)
        self.log.debug("Determined FBFUSE config: {}".format(self._fbf_sb_config))

        # New multicast setup
        # First we allocate all servers
        self._servers = []
        self._worker_config_map = {}
        while True:
            try:
                server = self._parent._server_pool.allocate(1)[0]
            except WorkerAllocationError:
                break
            else:
                self._servers.append(server)

        config_generator = ApsConfigGenerator(self._fbf_sb_config,
            self._data_rate_per_worker_sensor.value())
        self._worker_config_map = config_generator.allocate_groups(self._servers)
        message = "\n".join((
            "Could not allocate resources for capture of the following groups",
            "incoherent groups: {}".format(",".join(
                map(str, config_generator.remaining_incoherent_groups()))),
            "coherent groups: {}".format(",".join(
                map(str, config_generator.remaining_coherent_groups())))))
        self.log.warning(message)

        cb_data_rate = (self._fbf_sb_config["coherent-beam-multicast-groups-data-rate"]
            / self._fbf_sb_config["coherent-beam-count-per-group"])
        ib_data_rate = self._fbf_sb_config["incoherent-beam-multicast-group-data-rate"]
        cb_file_size = (cb_data_rate * DEFAULT_FILE_LENGTH) / 8
        ib_file_size = (ib_data_rate * DEFAULT_FILE_LENGTH) / 8
        self.log.info("CB filesize: {} bytes".format(cb_file_size))
        self.log.info("IB filesize: {} bytes".format(ib_file_size))

        # Get all common configuration parameters
        common_config = {
            "bandwidth": self._fbf_sb_config["bandwidth"],
            "centre-frequency": self._fbf_sb_config["centre-frequency"],
            "sample-clock": self._fbf_sb_config["bandwidth"] * 2,
        }
        common_config["sync-epoch"] = yield self._katportal_client.get_sync_epoch()

        common_coherent_config = {
            "heap-size": self._fbf_sb_config["coherent-beam-heap-size"],
            "idx1-step": self._fbf_sb_config["coherent-beam-idx1-step"],
            "nchans": self._fbf_sb_config["nchannels"] / self._fbf_sb_config["coherent-beam-fscrunch"],
            "nchans-per-heap": self._fbf_sb_config["coherent-beam-subband-nchans"],
            "sampling-interval": self._fbf_sb_config["coherent-beam-time-resolution"],
            "base-output-dir": "{}".format(self._base_output_dir),
            "filesize": cb_file_size
        }

        common_incoherent_config = {
            "heap-size": self._fbf_sb_config["incoherent-beam-heap-size"],
            "idx1-step": self._fbf_sb_config["incoherent-beam-idx1-step"],
            "nchans": self._fbf_sb_config["nchannels"] / self._fbf_sb_config["incoherent-beam-fscrunch"],
            "nchans-per-heap": self._fbf_sb_config["incoherent-beam-subband-nchans"],
            "sampling-interval": self._fbf_sb_config["incoherent-beam-time-resolution"],
            "base-output-dir": "{}".format(self._base_output_dir),
            "filesize": ib_file_size
        }

        configure_futures = []
        all_server_configs = {}
        for server, config in self._worker_config_map.items():
            server_config = {}
            if config.incoherent_groups():
                incoherent_config = deepcopy(common_config)
                incoherent_config.update(common_incoherent_config)
                incoherent_config["beam-ids"] = []
                incoherent_config["stream-indices"] = []
                incoherent_config["mcast-groups"] = []
                incoherent_config["mcast-port"] = 7147  # Where should this info come from?
                for beam in config.incoherent_beams():
                    incoherent_config["beam-ids"].append(beam)
                    incoherent_config["stream-indices"].append(int(beam.lstrip("ifbf")))
                incoherent_config["mcast-groups"].extend(map(str, config.incoherent_groups()))
                server_config["incoherent-beams"] = incoherent_config
            if config.coherent_groups():
                coherent_config = deepcopy(common_config)
                coherent_config.update(common_coherent_config)
                coherent_config["beam-ids"] = []
                coherent_config["stream-indices"] = []
                coherent_config["mcast-groups"] = []
                coherent_config["mcast-port"] = 7147  # Where should this info come from?
                for beam in config.coherent_beams():
                    coherent_config["beam-ids"].append(beam)
                    coherent_config["stream-indices"].append(int(beam.lstrip("cfbf")))
                coherent_config["mcast-groups"].extend(map(str, config.coherent_groups()))
                server_config["coherent-beams"] = coherent_config
            configure_futures.append(server.configure(server_config))
            all_server_configs[server] = server_config
            self.log.info("Configuration for server {}: {}".format(
                server, server_config))
        self._worker_configs_sensor.set_value(all_server_configs)

        failure_count = 0
        for future in configure_futures:
            try:
                yield future
            except Exception as error:
                log.error(
                    "Failed to configure server with error: {}".format(
                        str(error)))
                failure_count += 1

        if (failure_count == len(self._servers)) and not (len(self._servers) == 0):
            self._state_sensor.set_value(self.ERROR)
            self.log.info("Failed to prepare FBFUSE product")
            raise Exception("No APSUSE servers configured successfully")
        elif failure_count > 0:
            self.log.warning("{} APSUSE servers failed to configure".format(
                failure_count))

        # At this point we do the data-suspect tracking start
        self._coherent_beam_tracker = self._katportal_client.get_sensor_tracker(
            "fbfuse", "fbfmc_{}_coherent_beam_data_suspect".format(
                self._product_id))
        self._incoherent_beam_tracker = self._katportal_client.get_sensor_tracker(
            "fbfuse", "fbfmc_{}_incoherent_beam_data_suspect".format(
                self._product_id))
        self.log.info("Starting FBFUSE data-suspect tracking")
        yield self._coherent_beam_tracker.start()
        yield self._incoherent_beam_tracker.start()

        @coroutine
        def wait_for_on_target():
            self.log.info("Waiting for data-suspect flags to become False")
            self._state_interrupt.clear()
            try:
                yield self._coherent_beam_tracker.wait_until(
                    False, self._state_interrupt)
                yield self._incoherent_beam_tracker.wait_until(
                    False, self._state_interrupt)
            except Interrupt:
                self.log.debug("data-suspect tracker interrupted")
                pass
            else:
                self.log.info("data-suspect flags now False (on target)")
                try:
                    yield self.disable_all_writers()
                    yield self.enable_writers()
                except Exception:
                    log.exception("error")
                self._parent.ioloop.add_callback(wait_for_off_target)

        @coroutine
        def wait_for_off_target():
            self.log.info("Waiting for data-suspect flags to become True")
            self._state_interrupt.clear()
            try:
                yield self._coherent_beam_tracker.wait_until(
                    True, self._state_interrupt)
                yield self._incoherent_beam_tracker.wait_until(
                    True, self._state_interrupt)
            except Interrupt:
                self.log.debug("data-suspect tracker interrupted")
                pass
            else:
                self.log.info("data-suspect flags now True (off-target/retiling)")
                yield self.disable_all_writers()
                self._parent.ioloop.add_callback(wait_for_on_target)

        self._parent.ioloop.add_callback(wait_for_on_target)
        server_str = ",".join(["{s.hostname}:{s.port}".format(
            s=server) for server in self._servers])
        self._servers_sensor.set_value(server_str)
        self._state_sensor.set_value(self.CAPTURING)
        self.log.debug("Product moved to 'capturing' state")

    @coroutine
    def capture_stop(self):
        """
        @brief      Stops the beamformer servers streaming.

        @detail     This should only be called on a schedule block reconfiguration
                    if the same configuration persists between schedule blocks then
                    it is preferable to continue streaming rather than stopping and
                    starting again.
        """
        if not self.capturing and not self.error:
            return
        self._state_sensor.set_value(self.STOPPING)
        self._state_interrupt.set()

        if self._coherent_beam_tracker:
            yield self._coherent_beam_tracker.stop()
            self._coherent_beam_tracker = None
        if self._incoherent_beam_tracker:
            yield self._incoherent_beam_tracker.stop()
            self._incoherent_beam_tracker = None

        yield self.disable_all_writers()
        deconfigure_futures = []
        for server in self._worker_config_map.keys():
            self.log.info("Sending deconfigure to server {}".format(server))
            deconfigure_futures.append(server.deconfigure())

        for ii, future in enumerate(deconfigure_futures):
            try:
                yield future
            except Exception as error:
                server = self._worker_config_map.keys()[ii]
                self.log.exception("Failed to deconfigure worker {}: {}".format(
                    server, str(error)))

        yield self.reset_workers(self._worker_config_map.keys())
        self._parent._server_pool.deallocate(self._worker_config_map.keys())
        self.log.info("Deallocated all servers")
        self._worker_config_map = {}
        self._servers_sensor.set_value("")
        self._state_sensor.set_value(self.READY)

    @coroutine
    def reset_workers(self, workers, timeout=60.0):
        for server in workers:
            try:
                yield server.reset()
            except Exception as error:
                log.exception("Could not reset worker '{}' with error: {}".format(
                    str(server), str(error)))
        start = time.time()
        while time.time() < start + timeout:
            if all([server.is_connected() for server in workers]):
                raise Return()
            else:
                yield sleep(1)
        log.warning("Not all servers reset within {} second timeout".format(
            timeout))

