"""
Copyright (c) 2019 Jason Wu <jwu@mpifr-bonn.mpg.de>

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
import tempfile
import json
import os
import time
from astropy.time import Time
from subprocess import PIPE, Popen
from mpikat.effelsberg.edd.pipeline.dada import render_dada_header, make_dada_key_string
import shlex
import threading
import base64
#from katcp import Sensor
import tornado
import coloredlogs
import signal
import astropy.units as units
from optparse import OptionParser
from tornado.gen import coroutine
#from katcp import AsyncDeviceServer, Message, AsyncReply
from katcp import AsyncDeviceServer, Message, Sensor, AsyncReply
from katcp.kattypes import request, return_reply, Str

log = logging.getLogger("mpikat.effelsberg.edd.pipeline.pipeline")
log.setLevel('DEBUG')

RUN = True

#PIPELINES = {}

BLANK_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
PIPELINE_STATES = ["idle", "configuring", "ready",
                   "starting", "running", "stopping",
                   "deconfiguring", "error"]
#        "args": "-n 64 -b 67108864 -p -l",
CONFIG = {
    "base_output_dir": os.getcwd(),
    "dspsr_params":
    {
        "args": "-L 10 -r -minram 1024"
    },
    "dada_db_params":
    {
        "args": "-n 32 -b 409600000 -p -l",
        "key": "dada"
    },
    "dadc_db_params":
    {
        "args": "-n 32 -b 409600000 -p -l",
        "key": "dadc"
    },
    "dada_header_params":
    {
        "filesize": 32000000000,
        "telescope": "Effelsberg",
        "instrument": "asterix",
        "frequency_mhz": 1400.4,
        "receiver_name": "EDD",
        "mc_source": "239.2.1.154",
        "bandwidth": 800,
        "tsamp": 0.000625,
        "nbit": 8,
        "ndim": 1,
        "npol": 2,
        "nchan": 1,
        "resolution": 1,
        "dsb": 1
    }

}

"""
Central frequency of each band should be with BW of 162.5
239.2.1.150 2528.90625
239.2.1.151 2366.40625
239.2.1.152 2203.9075
239.2.1.153 2041.40625
239.2.1.154 1878.90625
239.2.1.155 1716.405
239.2.1.156 1553.9075
239.2.1.157 1391.40625
"""

sensors = {"ra": 123, "dec": -10, "source-name": "J1939+2134",
           "scannum": 0, "subscannum": 1}


class ExecuteCommand(object):

    def __init__(self, command, outpath=None, resident=False):
        self._command = command
        self._resident = resident
        self._outpath = outpath
        self.stdout_callbacks = set()
        self.stderr_callbacks = set()
        self.error_callbacks = set()
        self.fscrunch_callbacks = set()
        self.tscrunch_callbacks = set()
        self.profile_callbacks = set()
        self._monitor_threads = []
        self._process = None
        self._executable_command = None
        self._monitor_thread = None
        self._stdout = None
        self._stderr = None
        self._error = False
        self._finish_event = threading.Event()

        if not self._resident:
            self._finish_event.set()

        self._executable_command = shlex.split(self._command)

        if RUN:
            try:
                self._process = Popen(self._executable_command,
                                      stdout=PIPE,
                                      stderr=PIPE,
                                      bufsize=1,
                                      # shell=True,
                                      universal_newlines=True)
            except Exception as error:
                log.exception("Error while launching command: {}".format(
                    self._executable_command))
                self.error = True
            if self._process == None:
                self._error = True
            self._monitor_thread = threading.Thread(
                target=self._execution_monitor)
            self._stderr_monitor_thread = threading.Thread(
                target=self._stderr_monitor)
            self._monitor_thread.start()
            self._stderr_monitor_thread.start()
            if self._outpath is not None:
                self._png_monitor_thread = threading.Thread(
                    target=self._png_monitor)
                self._png_monitor_thread.start()

    def __del__(self):
        class_name = self.__class__.__name__

    def set_finish_event(self):
        if not self._finish_event.isSet():
            self._finish_event.set()

    def finish(self):
        if RUN:
            self._process.terminate()
            self._monitor_thread.join()
            self._stderr_monitor_thread.join()
            if self._outpath is not None:
                self._png_monitor_thread.join()

    def stdout_notify(self):
        for callback in self.stdout_callbacks:
            callback(self._stdout, self)

    @property
    def stdout(self):
        return self._stdout

    @stdout.setter
    def stdout(self, value):
        self._stdout = value
        self.stdout_notify()

    def stderr_notify(self):
        for callback in self.stderr_callbacks:
            callback(self._stderr, self)

    @property
    def stderr(self):
        return self._stderr

    @stderr.setter
    def stderr(self, value):
        self._stderr = value
        self.stderr_notify()

    def fscrunch_notify(self):
        for callback in self.fscrunch_callbacks:
            callback(self._fscrunch, self)

    @property
    def fscrunch(self):
        return self._fscrunch

    @fscrunch.setter
    def fscrunch(self, value):
        self._fscrunch = value
        self.fscrunch_notify()

    def tscrunch_notify(self):
        for callback in self.tscrunch_callbacks:
            callback(self._tscrunch, self)

    @property
    def tscrunch(self):
        return self._tscrunch

    @tscrunch.setter
    def tscrunch(self, value):
        self._tscrunch = value
        self.tscrunch_notify()

    def profile_notify(self):
        for callback in self.profile_callbacks:
            callback(self._profile, self)

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value):
        self._profile = value
        self.profile_notify()

    def error_notify(self):
        for callback in self.error_callbacks:
            callback(self._error, self)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value
        self.error_notify()

    def _execution_monitor(self):
        # Monitor the execution and also the stdout for the outside useage
        if RUN:
            while self._process.poll() == None:
                stdout = self._process.stdout.readline().rstrip("\n\r")
                if stdout != b"":
                    if (not stdout.startswith("heap")) & (not stdout.startswith("mark")) & (not stdout.startswith("[")) & (not stdout.startswith("-> parallel")) & (not stdout.startswith("-> sequential")):
                        self.stdout = stdout
                    # print self.stdout, self._command

            if not self._finish_event.isSet():
                # For the command which runs for a while, if it stops before
                # the event is set, that means that command does not successfully finished
                stdout = self._process.stdout.read()
                stderr = self._process.stderr.read()
                log.error(
                    "Process exited unexpectedly with return code: {}".format(self._process.returncode))
                log.error("exited unexpectedly, stdout = {}".format(stdout))
                log.error("exited unexpectedly, stderr = {}".format(stderr))
                log.error("exited unexpectedly, cmd = {}".format(self._command))
                self.error = True


    def _stderr_monitor(self):
        if RUN:
            while self._process.poll() == None:
                stderr = self._process.stderr.readline().rstrip("\n\r")
                if stderr != b"":
                    self.stderr = stderr
            if not self._finish_event.isSet():
                # For the command which runs for a while, if it stops before
                # the event is set, that means that command does not successfully finished
                stdout = self._process.stdout.read()
                stderr = self._process.stderr.read()
                log.error(
                    "Process exited unexpectedly with return code: {}".format(self._process.returncode))
                log.error("exited unexpectedly, stdout = {}".format(stdout))
                log.error("exited unexpectedly, stderr = {}".format(stderr))
                log.error("exited unexpectedly, cmd = {}".format(self._command))
                self.error = True

    def _png_monitor(self):
        if RUN:
            while self._process.poll() == None:
                log.debug("Accessing archive PNG files")
                try:
                    with open("{}/fscrunch.png".format(self._outpath), "rb") as imageFile:
                        self.fscrunch = base64.b64encode(imageFile.read())
                except Exception as error:
                    log.debug(error)
                    log.debug("fscrunch.png is not ready")
                try:
                    with open("{}/tscrunch.png".format(self._outpath), "rb") as imageFile:
                        self.tscrunch = base64.b64encode(imageFile.read())
                except Exception as error:
                    log.debug(error)
                    log.debug("tscrunch.png is not ready")
                try:
                    with open("{}/profile.png".format(self._outpath), "rb") as imageFile:
                        self.profile = base64.b64encode(imageFile.read())
                except Exception as error:
                    log.debug(error)
                    log.debug("profile.png is not ready")

                time.sleep(5)


class EddPulsarPipelineKeyError(Exception):
    pass


class EddPulsarPipelineError(Exception):
    pass


class EddPulsarPipeline(AsyncDeviceServer):
    """
    @brief Interface object which accepts KATCP commands

    """
    VERSION_INFO = ("mpikat-edd-api", 1, 0)
    BUILD_INFO = ("mpikat-edd-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]
    PIPELINE_STATES = ["idle", "configuring", "ready",
                       "starting", "running", "stopping",
                       "deconfiguring", "error"]

    def __init__(self, ip, port):
        """
        @brief Initialization of the EDDPulsarPipeline object

        @param ip       IP address of the server
        @param port     port of the EDDPulsarPipeline

        """
        super(EddPulsarPipeline, self).__init__(ip, port)
        # self.setup_sensors()
        self.ip = ip
        self._managed_sensors = []
        self.callbacks = set()
        self._state = "idle"
        #self._sensors = []
        self._volumes = ["/tmp/:/scratch/"]
        self._dada_key = None
        self._config = None
        self._source_config = None
        self._dspsr = None
        self._mkrecv_ingest_proc = None
        # self.setup_sensors()

    def notify(self):
        """@brief callback function."""
        for callback in self.callbacks:
            callback(self._state, self)

    @property
    def state(self):
        """@brief property of the pipeline state."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

    def add_pipeline_sensors(self):
        """
        @brief Add pipeline sensors to the managed sensors list

        """
        for sensor in self._pipeline_instance.sensors:
            log.debug("sensor name is {}".format(sensor))
            self.add_sensor(sensor)
            self._managed_sensors.append(sensor)
        self.mass_inform(Message.inform('interface-changed'))

    def remove_pipeline_sensors(self):
        """
        @brief Remove pipeline sensors from the managed sensors list

        """
        for sensor in self._managed_sensors:
            self.remove_sensor(sensor)
        self._managed_sensors = []
        self.mass_inform(Message.inform('interface-changed'))

    def state_change(self, state, callback):
        """
        @brief callback function for state changes

        @parma callback object return from the callback function from the pipeline
        """
        log.info('New state of the pipeline is {}'.format(str(state)))
        self._pipeline_sensor_status.set_value(str(state))

    @coroutine
    def start(self):
        super(EddPulsarPipeline, self).start()

    @coroutine
    def stop(self):
        """Stop PafWorkerServer server"""
        # if self._pipeline_sensor_status.value() == "ready":
        #    log.info("Pipeline still running, stopping pipeline")
       # yield self.deconfigure()
        yield super(EddPulsarPipeline, self).stop()

    def setup_sensors(self):
        """
        @brief Setup monitoring sensors
        """
        self._device_status = Sensor.discrete(
            "device-status",
            "Health status of PafWorkerServer",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)

        self._pipeline_sensor_name = Sensor.string("pipeline-name",
                                                   "the name of the pipeline", "")
        self.add_sensor(self._pipeline_sensor_name)

        self._pipeline_sensor_status = Sensor.discrete(
            "pipeline-status",
            description="Status of the pipeline",
            params=self.PIPELINE_STATES,
            default="idle",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._pipeline_sensor_status)

        self._tscrunch = Sensor.string(
            "tscrunch_PNG",
            description="tscrunch png",
            default=BLANK_IMAGE,
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._tscrunch)

        self._fscrunch = Sensor.string(
            "fscrunch_PNG",
            description="fscrunch png",
            default=BLANK_IMAGE,
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._fscrunch)

        self._profile = Sensor.string(
            "profile_PNG",
            description="pulse profile png",
            default=BLANK_IMAGE,
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._profile)

        self._central_freq = Sensor.string(
            "_central_freq",
            description="_central_freq",
            default="N/A",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._central_freq)

        self._source_name_sensor = Sensor.string(
            "source_name",
            description="target name",
            default="N/A",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._source_name_sensor)

        self._nchannels = Sensor.string(
            "_nchannels",
            description="_nchannels",
            default="N/A",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._nchannels)

        self._nbins = Sensor.string(
            "_nbins",
            description="_nbins",
            default="N/A",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._nbins)

        self._time_processed = Sensor.string(
            "_time_processed",
            description="_time_processed",
            default="N/A",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._time_processed)

    @property
    def sensors(self):
        return self._sensors

    def _decode_capture_stdout(self, stdout, callback):
        log.debug('{}'.format(str(stdout)))

    def _save_capture_stdout(self, stdout, callback):
        with open("{}.par".format(self._source_config["source-name"]), "a") as file:
            file.write('{}\n'.format(str(stdout)))

    def _handle_execution_returncode(self, returncode, callback):
        log.debug(returncode)

    def _handle_execution_stderr(self, stderr, callback):
        if bool(stderr[:8] == "Finished") & bool("." not in stderr):
            self._time_processed.set_value(stderr)
            log.info(stderr)
        if bool(stderr[:8] != "Finished"):
            log.info(stderr)

    def _add_tscrunch_to_sensor(self, png_blob, callback):
        self._tscrunch.set_value(png_blob)

    def _add_fscrunch_to_sensor(self, png_blob, callback):
        self._fscrunch.set_value(png_blob)

    def _add_profile_to_sensor(self, png_blob, callback):
        self._profile.set_value(png_blob)

    @request(Str())
    @return_reply()
    def request_configure(self, req, config_json):
        """
        @brief      Configure pipeline
        @param      config_json     A dictionary containing configuration information.
                                    The dictionary should have a form similar to:
                                    @code
                                    {
                                    "mode":"DspsrPipelineSrxdev",
                                    "mc_source":"239.2.1.154",
                                    "central_freq":"1400.4"
                                    }
                                    @endcode
        """
        @coroutine
        def configure_wrapper():
            try:
                yield self.configure_pipeline(config_json)
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
                self._pipeline_sensor_status.set_value("ready")
        self.ioloop.add_callback(configure_wrapper)
        raise AsyncReply

    @coroutine
    def configure_pipeline(self, config_json):
        try:
            config_dict = json.loads(config_json)
            pipeline_name = config_dict["mode"]
            log.debug("Pipeline name = {}".format(pipeline_name))
        except KeyError as error:
            msg = "Error getting the pipeline name from config_json: {}".format(
                str(error))
            log.error(msg)
            raise EddPulsarPipelineKeyError(msg)
        log.info("Configuring pipeline {}".format(
            self._pipeline_sensor_name.value()))

        try:
            config = json.loads(config_json)
        except Exception as error:
            log.info("Cannot load config json :{}".format(error))

        try:
            log.debug("Unpacked config: {}".format(config))
            self._pipeline_config = json.loads(config_json)
            self._config = CONFIG
            self._dada_key = "dada"
            self._dadc_key = "dadc"
        except Exception as error:
            log.info("Cannot unpack config json :{}".format(error))

        try:
            log.debug("Deconfiguring pipeline before configuring")
            self.deconfigure()
        except Exception as error:
            raise EddPulsarPipelineError(str(error))

        try:
            self._pipeline_sensor_name.set_value(pipeline_name)
            log.info("Creating DADA buffer for mkrecv")
            cmd = "numactl -m 1 dada_db -k {key} {args}".format(key=self._dada_key,
                                                                args=self._config["dada_db_params"]["args"])
            # cmd = "dada_db -k {key} {args}".format(**
            #                                       self._config["dada_db_params"])
            log.debug("Running command: {0}".format(cmd))
            self._create_ring_buffer = ExecuteCommand(
                cmd, outpath=None, resident=False)
            self._create_ring_buffer.stdout_callbacks.add(
                self._decode_capture_stdout)
            self._create_ring_buffer._process.wait()
        except Exception as error:
            raise EddPulsarPipelineError(str(error))
        try:
            log.info("Creating DADA buffer for EDDPolnMerge")
            cmd = "numactl -m 1 dada_db -k {key} {args}".format(key=self._dadc_key,
                                                                args=self._config["dadc_db_params"]["args"])
            # cmd = "dada_db -k {key} {args}".format(**
            #                                       self._config["dada_db_params"])
            log.debug("Running command: {0}".format(cmd))
            self._create_transpose_ring_buffer = ExecuteCommand(
                cmd, outpath=None, resident=False)
            self._create_transpose_ring_buffer.stdout_callbacks.add(
                self._decode_capture_stdout)
            self._create_transpose_ring_buffer._process.wait()
        except Exception as error:
            raise EddPulsarPipelineError(str(error))
        else:
            self.state = "ready"
            log.info("Pipeline instance {} configured".format(
                self._pipeline_sensor_name.value()))

    @request(Str())
    @return_reply(Str())
    def request_start(self, req, config_json):
        """
        @brief      Start pipeline
        @param      config_json     A dictionary containing configuration information.
                                    The dictionary should have a form similar to:
                                    @code
                                    {
                                    "source-name":"J1022+1001",
                                    "ra":"123.4",
                                    "dec":"-20.1"
                                    }
                                    @endcode
        """
        @coroutine
        def start_wrapper():
            try:
                yield self.start_pipeline(config_json)
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
                self._pipeline_sensor_status.set_value("running")
        self.ioloop.add_callback(start_wrapper)
        raise AsyncReply

    @coroutine
    def start_pipeline(self, config_json):
        """@brief start the dspsr instance then turn on dada_junkdb instance."""
        # if self.state == "ready":
        #    self.state = "starting"
        try:
            self._source_config = json.loads(config_json)
            log.info("Unpacked config: {}".format(self._source_config))
            self.frequency_mhz, self.bandwidth = self._pipeline_config[
                "central_freq"], self._pipeline_config["bandwidth"]
            self._central_freq.set_value(str(self.frequency_mhz))
            header = self._config["dada_header_params"]
            header["ra"], header["dec"], header["key"], header["mc_source"], header["frequency_mhz"], header["bandwidth"] = self._source_config[
                "ra"], self._source_config["dec"], self._dada_key, self._pipeline_config[
                "mc_source"], self.frequency_mhz, self.bandwidth
            self.source_name, self.nchannels, self.nbins = self._source_config[
                "source-name"], self._source_config["nchannels"], self._source_config["nbins"]
            self._source_name_sensor.set_value(self.source_name)
            self._nchannels.set_value(self.nchannels)
            self._nbins.set_value(self.nbins)
        except KeyError as error:
            msg = "Key error from reading config_json: {}".format(
                str(error))
            log.error(msg)
            raise EddPulsarPipelineKeyError(msg)

        #cpu_numbers = self._pipeline_config["cpus"]
        #cuda_number = self._pipeline_config["cuda"]
        cpu_numbers = "30,31"
        cuda_number = "1"
        try:
            header["sync_time"] = self._source_config["sync_time"]
            header["sample_clock"] = self._source_config["sample_clock"]
        except:
            pass
        ########NEED TO PUT IN THE LOGIC FOR _R here#############
        try:
            self.source_name = self.source_name.split("_")[0]
        except Exception as error:
            raise EddPulsarPipelineError(str(error))
        header["source_name"] = self.source_name
        header["obs_id"] = "{0}_{1}".format(
            sensors["scannum"], sensors["subscannum"])
        tstr = Time.now().isot.replace(":", "-")
        ####################################################
        #SETTING UP THE INPUT AND SCRUNCH DATA DIRECTORIES #
        ####################################################
        try:
            in_path = os.path.join("/media/scratch/jason/dspsr_output/", self.source_name,
                                   str(self.frequency_mhz), tstr, "raw_data")
            out_path = os.path.join(
                "/media/scratch/jason/dspsr_output/", self.source_name, str(self.frequency_mhz), tstr, "combined_data")
            self.out_path = out_path
            log.debug("Creating directories")
            cmd = "mkdir -p {}".format(in_path)
            log.debug("Command to run: {}".format(cmd))
            log.debug("Current working directory: {}".format(os.getcwd()))
            self._create_workdir_in_path = ExecuteCommand(
                cmd, outpath=None, resident=False)
            self._create_workdir_in_path.stdout_callbacks.add(
                self._decode_capture_stdout)
            self._create_workdir_in_path._process.wait()
            cmd = "mkdir -p {}".format(out_path)
            log.debug("Command to run: {}".format(cmd))
            log.info("Createing data directory {}".format(self.out_path))
            self._create_workdir_out_path = ExecuteCommand(
                cmd, outpath=None, resident=False)
            self._create_workdir_out_path.stdout_callbacks.add(
                self._decode_capture_stdout)
            self._create_workdir_out_path._process.wait()
            os.chdir(in_path)
            log.debug("Change to workdir: {}".format(os.getcwd()))
            log.debug("Current working directory: {}".format(os.getcwd()))
        except Exception as error:
            yield self.stop_pipeline()
            raise EddPulsarPipelineError(str(error))

        ####################################################
        #CREATING THE PARFILE WITH PSRCAT                  #
        ####################################################
        try:
            cmd = "psrcat -E {source_name}".format(
                source_name=self.source_name)
            log.debug("Command to run: {}".format(cmd))
            self.psrcat = ExecuteCommand(cmd, outpath=None, resident=False)
            self.psrcat.stdout_callbacks.add(
                self._save_capture_stdout)
            self.psrcat.stderr_callbacks.add(
                self._handle_execution_stderr)
        except Exception as error:
            yield self.stop_pipeline()
            raise EddPulsarPipelineError(str(error))
        # time.sleep(3)
        ####################################################
        #CREATING THE PREDICTOR WITH TEMPO2                #
        ####################################################
        cmd = 'tempo2 -f {}.par -pred "Effelsberg {} {} {} {} 8 2 3599.999999999"'.format(
            self.source_name, Time.now().mjd - 2, Time.now().mjd + 2, float(self._pipeline_config["central_freq"]) - 1.0, float(self._pipeline_config["central_freq"]) + 1.0)
        log.debug("Command to run: {}".format(cmd))
        self.tempo2 = ExecuteCommand(cmd, outpath=None, resident=False)
        self.tempo2.stdout_callbacks.add(
            self._decode_capture_stdout)
        self.tempo2.stderr_callbacks.add(
            self._handle_execution_stderr)
        ####################################################
        #CREATING THE DADA HEADERFILE                      #
        ####################################################
        dada_header_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="edd_dada_header_",
            suffix=".txt",
            dir=os.getcwd(),
            delete=False)
        log.debug(
            "Writing dada header file to {0}".format(
                dada_header_file.name))
        header_string = render_dada_header(header)
        dada_header_file.write(header_string)
        dada_key_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="dada_keyfile_",
            suffix=".key",
            dir=os.getcwd(),
            delete=False)
        log.debug("Writing dada key file to {0}".format(
            dada_key_file.name))
        key_string = make_dada_key_string(self._dadc_key)
        dada_key_file.write(make_dada_key_string(self._dadc_key))
        log.debug("Dada key file contains:\n{0}".format(key_string))
        dada_header_file.close()
        dada_key_file.close()
        time.sleep(5)
        ####################################################
        #STARTING DSPSR                                    #
        ####################################################
        cmd = "numactl -m 1 dspsr {args} {nchan} {nbin} -cpu {cpus} -cuda {cuda_number} -P {predictor} -E {parfile} {keyfile}".format(
            args=self._config["dspsr_params"]["args"],
            nchan="-F {}:D".format(self.nchannels),
            nbin="-b {}".format(self.nbins),
            predictor="{}/t2pred.dat".format(in_path),
            parfile="{}/{}.par".format(in_path, self.source_name),
            cpus=cpu_numbers,
            cuda_number=cuda_number,
            keyfile=dada_key_file.name)
        #cmd = "numactl -m 1 dbnull -k dadc"
        log.debug("Running command: {0}".format(cmd))
        log.info("Staring DSPSR")
        self._dspsr = ExecuteCommand(cmd, outpath=None, resident=True)
        self._dspsr.stdout_callbacks.add(
            self._decode_capture_stdout)
        self._dspsr.stderr_callbacks.add(
            self._handle_execution_stderr)
        # time.sleep(5)
        ####################################################
        #STARTING EDDPolnMerge                             #
        ####################################################
        cmd = "numactl -m 1 taskset -c 29 edd_merge"
        log.debug("Running command: {0}".format(cmd))
        log.info("Staring EDDPolnMerge")
        self._polnmerge_proc = ExecuteCommand(
            cmd, outpath=None, resident=True)
        self._polnmerge_proc.stdout_callbacks.add(
            self._decode_capture_stdout)
        time.sleep(5)
        ####################################################
        #STARTING MKRECV                                   #
        ####################################################
        cmd = "numactl -m 1 taskset -c 18-28 mkrecv_nt --header {} --dada-mode 4 --quiet".format(
            dada_header_file.name)
        log.debug("Running command: {0}".format(cmd))
        log.info("Staring MKRECV")
        self._mkrecv_ingest_proc = ExecuteCommand(
            cmd, outpath=None, resident=True)
        self._mkrecv_ingest_proc.stdout_callbacks.add(
            self._decode_capture_stdout)

        ####################################################
        #STARTING ARCHIVE MONITOR                          #
        ####################################################
        cmd = "python /src/mpikat/mpikat/effelsberg/edd/pipeline/archive_directory_monitor.py -i {} -o {}".format(
            in_path, out_path)
        log.debug("Running command: {0}".format(cmd))
        log.info("Staring archive monitor")
        self._archive_directory_monitor = ExecuteCommand(
            cmd, outpath=out_path, resident=True)
        self._archive_directory_monitor.stdout_callbacks.add(
            self._decode_capture_stdout)
        self._archive_directory_monitor.fscrunch_callbacks.add(
            self._add_fscrunch_to_sensor)
        self._archive_directory_monitor.tscrunch_callbacks.add(
            self._add_tscrunch_to_sensor)
        self._archive_directory_monitor.profile_callbacks.add(
            self._add_profile_to_sensor)

        # except Exception as error:
        #    msg = "Couldn't start pipeline server {}".format(str(error))
        #    log.error(msg)
        #    raise EddPulsarPipelineError(msg)
        # else:
        log.info("Starting pipeline {}".format(
            self._pipeline_sensor_name.value()))

    @request()
    @return_reply(Str())
    def request_stop(self, req):
        """
        @brief      Stop pipeline

        """
        @coroutine
        def stop_wrapper():
            try:
                yield self.stop_pipeline()
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
                self._pipeline_sensor_status.set_value("ready")
        self.ioloop.add_callback(stop_wrapper)
        raise AsyncReply

    @coroutine
    def stop_pipeline(self):
        """@brief stop the dada_junkdb and dspsr instances."""
        try:
            log.debug("Stopping")
            self._timeout = 10
            # process = [self._mkrecv_ingest_proc,
            #          self._dspsr, self._archive_directory_monitor]
            process = [self._mkrecv_ingest_proc,
                       self._polnmerge_proc, self._dspsr, self._archive_directory_monitor]
            for proc in process:
                proc.set_finish_event()
                proc.finish()
                log.debug(
                    "Waiting {} seconds for proc to terminate...".format(self._timeout))
                now = time.time()
                while time.time() - now < self._timeout:
                    retval = proc._process.poll()
                    if retval is not None:
                        log.debug(
                            "Returned a return value of {}".format(retval))
                        break
                    else:
                        time.sleep(0.5)
                else:
                    log.warning(
                        "Failed to terminate proc in alloted time")
                    log.info("Killing process")
                    proc._process.kill()
        except Exception as error:
            msg = "Couldn't stop pipeline {}".format(str(error))
            log.error(msg)
            raise EddPulsarPipelineError(msg)
        else:
            log.info("Pipeline Stopped {}".format(
                self._pipeline_sensor_name.value()))

    @request()
    @return_reply(Str())
    def request_deconfigure(self, req):
        """
        @brief      Deconfigure pipeline

        """
        @coroutine
        def deconfigure_wrapper():
            if self._pipeline_sensor_status.value == 'running':
                yield self.stop_pipeline()
            try:
                yield self.deconfigure()
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
                self._pipeline_sensor_status.set_value("idle")
        self.ioloop.add_callback(deconfigure_wrapper)
        raise AsyncReply

    @coroutine
    def deconfigure(self):
        """@brief deconfigure the dspsr pipeline."""
        log.info("Deconfiguring pipeline {}".format(
            self._pipeline_sensor_name.value()))
        log.debug("Destroying dada buffers")
        try:
            self.remove_pipeline_sensors()
            cmd = "dada_db -d -k {0}".format(self._dada_key)
            log.debug("Running command: {0}".format(cmd))
            self._destory_ring_buffer = ExecuteCommand(
                cmd, outpath=None, resident=False)
            self._destory_ring_buffer.stdout_callbacks.add(
                self._decode_capture_stdout)
            self._destory_ring_buffer._process.wait()

            cmd = "dada_db -d -k {0}".format(self._dadc_key)
            log.debug("Running command: {0}".format(cmd))
            self._destory_merge_buffer = ExecuteCommand(
                cmd, outpath=None, resident=False)
            self._destory_merge_buffer.stdout_callbacks.add(
                self._decode_capture_stdout)
            self._destory_merge_buffer._process.wait()
            self._fscrunch.set_value(BLANK_IMAGE)
            self._tscrunch.set_value(BLANK_IMAGE)
            self._profile.set_value(BLANK_IMAGE)

        except Exception as error:
            msg = "Couldn't deconfigure pipeline {}".format(str(error))
            log.error(msg)
            raise EddPulsarPipelineError(msg)
        else:
            log.info("Deconfigured pipeline {}".format(
                self._pipeline_sensor_name.value()))
            self._pipeline_sensor_name.set_value("")


@coroutine
def on_shutdown(ioloop, server):
    log.info('Shutting down server')
    if server._pipeline_sensor_status.value() == "running":
        log.info("Pipeline still running, stopping pipeline")
        yield server.stop_pipeline()
        time.sleep(10)
    if server._pipeline_sensor_status.value() != "idle":
        log.info("Pipeline still configured, deconfiguring pipeline")
        yield server.deconfigure()
    yield server.stop()
    ioloop.stop()


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-H', '--host', dest='host', type=str,
                      help='Host interface to bind to', default="127.0.0.1")
    parser.add_option('-p', '--port', dest='port', type=long,
                      help='Port number to bind to', default=5000)
    parser.add_option('', '--log_level', dest='log_level', type=str,
                      help='logging level', default="INFO")
    (opts, args) = parser.parse_args()
    logging.getLogger().addHandler(logging.NullHandler())
    logger = logging.getLogger('mpikat')
    coloredlogs.install(
        fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level=opts.log_level.upper(),
        logger=logger)
    logging.getLogger('katcp').setLevel('INFO')
    ioloop = tornado.ioloop.IOLoop.current()
    log.info("Starting EddPulsarPipeline instance")
    server = EddPulsarPipeline(opts.host, opts.port)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))

    def start_and_display():
        server.start()
        log.info(
            "Listening at {0}, Ctrl-C to terminate server".format(server.bind_address))

    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    main()
