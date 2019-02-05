import signal
import logging
import tempfile
import coloredlogs
import tornado
import datetime
from tornado import gen
import os
import time
import shutil
from datetime import datetime
from subprocess import check_output, PIPE, Popen
from katcp import AsyncDeviceServer, Sensor, ProtocolFlags, AsyncReply
from katcp.kattypes import request, return_reply, Int, Str, Discrete, Float
from mpikat.effelsberg.edd.pipeline.dada import render_dada_header, make_dada_key_string
import shlex
log = logging.getLogger("mpikat.effelsberg.edd.pipeline.pipeline")
log.setLevel('DEBUG')
#
# NOTE: For this to run properly the host /tmp/
# directory should be mounted onto the launching container.
# This is needed as docker doesn't currently support
# container to container file copies.
#
RUN = True

PIPELINES = {}

PIPELINE_STATES = ["idle", "configuring", "ready",
                   "starting", "running", "stopping",
                   "deconfiguring", "error"]

CONFIG = {
    "base_output_dir": os.getcwd(),
    "dspsr_params":
    {
        "args": "-cpu 2,3 -L 10 -r -F 256:D -cuda 0,0 -minram 1024"
    },
    "dada_db_params":
    {
        "args": "-n 8 -b 1280000000 -p -l",
        "key": "dada"
    },
    "dada_header_params":
    {
        "filesize": 32000000000,
        "telescope": "Effelsberg",
        "instrument": "asterix",
        "frequency_mhz": 1370,
        "receiver_name": "P200-3",
        "bandwidth": 320,
        "tsamp": 0.00156250,
        "nbit": 8,
        "ndim": 1,
        "npol": 2,
        "nchan": 1,
        "resolution": 1,
        "dsb": 1
    }

}

sensors = {"ra": 123, "dec": -10, "source-name": "J1939+2134",
           "scannum": 0, "subscannum": 1, "timestamp": str(datetime.now().time())}

DESCRIPTION = """
This pipeline captures data from the network and passes it to a dada
ring buffer for processing by DSPSR
""".lstrip()

def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        log.debug('got line from subprocess: %r', line)

def register_pipeline(name):
    def _register(cls):
        PIPELINES[name] = cls
        return cls
    return _register

def safe_popen(cmd, *args, **kwargs):
    if RUN == True:
        process = Popen(shlex.split(cmd), stdout=PIPE)
    else:
        process = None
    return process 

@register_pipeline("DspsrPipeline")
class Mkrecv2Db2Dspsr(object):

    def __del__(self):
        class_name = self.__class__.__name__

    def notify(self):
        for callback in self.callbacks:
            callback(self._state, self)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
        
    def __init__(self):
        self.callbacks = set()
        self._state = "idle" 
        self._volumes = ["/tmp/:/scratch/"]
        self._dada_key = None
        self._config = None
        self._dspsr = None
        self._mkrecv_ingest_proc = None


    @gen.coroutine
    def configure(self):
        self._config = CONFIG
        self._dada_key = CONFIG["dada_db_params"]["key"]
        try:
            self.deconfigure()
        except Exception:
            pass
        cmd = "dada_db -k {key} {args}".format(**
                                               self._config["dada_db_params"])
        log.debug("Running command: {0}".format(cmd))

        self._create_ring_buffer = safe_popen(cmd, stdout=PIPE)
        with self._create_ring_buffer.stdout:
            log_subprocess_output(process.stdout)
        self._create_ring_buffer.wait()
        log.debug(output)
        self.state = "ready"

    @gen.coroutine
    def start(self):
        self.state = "running"
        header = self._config["dada_header_params"]
        header["ra"] = sensors["ra"]
        header["dec"] = sensors["dec"]
        source_name = sensors["source-name"]
        try:
            source_name = source_name.split("_")[0]
        except Exception:
            pass
        header["source_name"] = source_name
        header["obs_id"] = "{0}_{1}".format(
            sensors["scannum"], sensors["subscannum"])
        tstr = sensors["timestamp"].replace(":", "-")  # to fix docker bug
        out_path = os.path.join("/beegfs/jason/", source_name, tstr)
        log.debug("Creating directories")
        cmd = "mkdir -p {}".format(out_path)
        log.debug("Command to run: {}".format(cmd))
        log.debug("Current working directory: {}".format(os.getcwd()))
        process = safe_popen(cmd, stdout=PIPE)
        process.wait()
        os.chdir(out_path)
        log.debug("Change to workdir: {}".format(os.getcwd()))
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
        #log.debug("Header file contains:\n{0}".format(header_string))
        dada_key_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="dada_keyfile_",
            suffix=".key",
            dir=os.getcwd(),
            delete=False)
        log.debug("Writing dada key file to {0}".format(dada_key_file.name))
        key_string = make_dada_key_string(self._dada_key)
        dada_key_file.write(make_dada_key_string(self._dada_key))
        log.debug("Dada key file contains:\n{0}".format(key_string))
        dada_header_file.close()
        dada_key_file.close()
        cmd = "dspsr {args} -N {source_name} {keyfile}".format(
            args=self._config["dspsr_params"]["args"],
            source_name=source_name,
            keyfile=dada_key_file.name)
        log.debug("Running command: {0}".format(cmd))
        self._dspsr = safe_popen(cmd, stdout=PIPE)
        
        ###################
        # Start up MKRECV
        ###################
        # if RUN is True:
        #self._mkrecv_ingest_proc = Popen(["mkrecv","--config",self._mkrecv_config_filename], stdout=PIPE, stderr=PIPE)

        cmd = "dada_junkdb -k {0} -b 320000000000 -r 1024 -g {1}".format(
            self._dada_key,
            dada_header_file.name)
        log.debug("running command: {}".format(cmd))
        self._dada_junkdb = safe_popen(cmd, stdout=PIPE)
        self.running_process_dada_junkdb = yield self._dada_junkdb
        self.running_process_dspsr = yield self._dspsr
        raise gen.Return(dada_junkdb.body)
    
    @gen.coroutine
    def stop(self):
        log.debug("Stopping")
        try:
            self._dspsr.terminate()
            self._dada_junkdb.terminate()
        except Exception:
            self._dspsr.kill()
            self._dada_junkdb.kill()
            self.deconfigure()
        self.state = "ready"

    def deconfigure(self):
        self.state = "idle"
        log.debug("Destroying dada buffer")
        cmd = "dada_db -d -k {0}".format(self._dada_key)
        log.debug("Running command: {0}".format(cmd))
        #args = shlex.split(cmd)
        process = safe_popen(cmd, stdout=PIPE)
        process.wait()

        #log.debug("Sending SIGTERM to MKRECV process")
        #    self._mkrecv_ingest_proc.terminate()
        #    self._mkrecv_timeout = 10.0
        #    log.debug("Waiting {} seconds for MKRECV to terminate...".format(self._mkrecv_timeout))
        #    now = time.time()
        #    while time.time()-now < self._mkrecv_timeout:
        #        retval = self._mkrecv_ingest_proc.poll()
        #        if retval is not None:
        #            log.info("MKRECV returned a return value of {}".format(retval))
        #            break
        #        else:
        #            yield sleep(0.5)
        #    else:
        #        log.warning("MKRECV failed to terminate in alloted time")
        #        log.info("Killing MKRECV process")
        #        self._mkrecv_ingest_proc.kill()


def main():
    print "\nCreate pipeline ...\n"
    logging.info("Starting pipeline instance")
    server = Mkrecv2Db2Dspsr()
    server.configure()
    server.start()
    # server.stop()
    server.deconfigure()

if __name__ == "__main__":
    main()
