"""
Copyright (c) 2020 Ewan Barr <ebarr@mpifr-bonn.mpg.de>

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
from tornado.ioloop import PeriodicCallback
from mpikat.utils.process_tools import ManagedProcess
from mpikat.utils.unix_socket import UDSClient

log = logging.getLogger('mpikat.fbfuse_transient_buffer')


class TransientBuffer(object):
    # manage unix socket connection

    # Start UDS Client
    # Make message
    def __init__(self,
                 dada_key,
                 nantennas,
                 partition_nchans,
                 partition_cfreq,
                 partition_bw,
                 total_nchans,
                 socket_name="/tmp/tb_trigger.sock",
                 output_dir="/transient_buffer_output/",
                 fill_level=0.8):
        self._dada_key = dada_key
        self._nantennas = nantennas
        self._nchans = partition_nchans
        self._cfreq = partition_cfreq
        self._bw = partition_bw
        self._total_nchans = total_nchans
        self._socket_name = socket_name
        self._output_dir = output_dir
        self._proc = None
        self._proc_mon = None
        self._running = False
        self._fill_level = fill_level

    def start(self, core=0):
        args = [
            "taskset", "-c", str(core),
            "bufferdump_to_file",
            "--input_key", self._dada_key,
            "--socket_name", self._socket_name,
            "--max_fill_level", self._fill_level,
            "--nantennas", self._nantennas,
            "--subband_nchannels", self._nchans,
            "--nchannels", self._total_nchans,
            "--centre_freq", self._cfreq,
            "--bandwidth", self._bw,
            "--outdir", self._output_dir,
            "--log_level", "info"]
        # Create SPEAD receiver for incoming antenna voltages
        self._proc = ManagedProcess(args)
        self._running = True

        def exit_check_callback():
            if not self._proc.is_alive():
                log.error("Transient buffer dumper exited unexpectedly")
        self._proc_mon = PeriodicCallback(exit_check_callback, 1000)
        self._proc_mon.start()

    def stop(self):
        if self._proc_mon:
            self._proc_mon.stop()
            self._proc_mon = None
        if self._proc:
            self._running = False
            self._proc.terminate()
            self._proc = None

    def trigger(self, utc_start, utc_end, dm, ref_freq, trigger_id):
        """
        Send a trigger message to the tb_capture socket

        :param      utc_start:   The utc start in UNIX time
        :param      utc_end:     The utc end in UNIX time
        :param      dm:          The dispersion measure in pccm
        :param      ref_freq:    The reference frequency in Hz
        :param      trigger_id:  The trigger identifier
        """
        if not self._running:
            log.error("Cannot trigger TB capture as no process active")
            return
        message = {
            "utc_start": utc_start,
            "utc_end": utc_end,
            "dm": dm,
            "reference_freq": ref_freq,
            "trigger_id": trigger_id
        }
        message_json = json.dumps(message)
        client = UDSClient(self._socket_name)
        client.send(message_json)
        try:
            response = json.loads(client.recv())
            log.info(response)
        except Exception as error:
            log.exception("Failed to trigger TB with error: {}".format(str(error)))
        finally:
            client.close()
